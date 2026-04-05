"""
Ticket routes: submit tickets, view history, retrieve individual tickets.
All routes require a valid JWT.

Multi-turn flow:
  - First message: no ticket_id in body → create new conversation doc, return ticket_id
  - Follow-up:     ticket_id present → look up doc, concatenate all prior user text
                   when previous strategy was clarify, run pipeline, append messages
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from db.connection import get_db
from db.models import build_conversation_doc
from api.auth_utils import get_current_user
from api.pipeline import run_pipeline

router = APIRouter()


class TicketRequest(BaseModel):
    text: str
    ticket_id: Optional[str] = None   # present on follow-up turns


@router.post("/ticket/new")
async def new_ticket(req: TicketRequest, user_id: str = Depends(get_current_user)):
    db  = get_db()
    now = datetime.utcnow()

    # ── Follow-up turn ────────────────────────────────────────────────────────
    if req.ticket_id:
        existing = await db.tickets.find_one(
            {"_id": req.ticket_id, "user_id": user_id}
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Ticket not found")

        previous_strategy = existing.get("rl_decision", {}).get("strategy", "")

        # Count how many turns have happened so far (each turn = 2 messages)
        turn_count = len(existing.get("messages", [])) // 2 + 1  # +1 for current turn

        if previous_strategy == "clarify":
            prior_user_texts = [
                m["text"] for m in existing.get("messages", [])
                if m.get("role") == "user"
            ]
            combined_text = " ".join(prior_user_texts + [req.text])
        else:
            combined_text = req.text

        result = run_pipeline(combined_text, user_id, turn_count=turn_count)

        assistant_text = (
            result.get("generator_output")
            or result.get("clarification_question")
            or result.get("response")
            or ""
        )

        rag_top_k    = result.get("rag", {}).get("top_k", [])
        sla_snapshot = result.get("sla", {})

        await db.tickets.update_one(
            {"_id": req.ticket_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role":      "user",
                                "text":      req.text,
                                "timestamp": now,
                            },
                            {
                                "role":         "assistant",
                                "text":         assistant_text,
                                "strategy":     result["rl"]["strategy"],
                                "action":       result["rl"]["action"],
                                "rag_top_k":    rag_top_k,
                                "sla_snapshot": sla_snapshot,
                                "timestamp":    now,
                            },
                        ]
                    }
                },
                "$set": {
                    "nlp_result":         result.get("nlp", {}),
                    "rag_result":         result.get("rag", {}),
                    "rl_decision":        result.get("rl", {}),
                    "sla_result":         sla_snapshot,
                    "generated_response": result.get("response"),
                    "resolved":           result.get("resolved", False),
                    "final_result": {
                        "strategy":       result["rl"]["strategy"],
                        "action":         result["rl"]["action"],
                        "top_similarity": result["rag"]["top_similarity"],
                    },
                },
            },
        )

        return {**result, "ticket_id": req.ticket_id}

    # ── First turn ────────────────────────────────────────────────────────────
    result = run_pipeline(req.text, user_id, turn_count=1)

    assistant_text = (
        result.get("generator_output")
        or result.get("clarification_question")
        or result.get("response")
        or ""
    )

    rag_top_k    = result.get("rag", {}).get("top_k", [])
    sla_snapshot = result.get("sla", {})

    doc = build_conversation_doc(
        user_id=user_id,
        raw_text=req.text,
        first_user_msg={
            "role":      "user",
            "text":      req.text,
            "timestamp": now,
        },
        first_assistant_msg={
            "role":         "assistant",
            "text":         assistant_text,
            "strategy":     result["rl"]["strategy"],
            "action":       result["rl"]["action"],
            "rag_top_k":    rag_top_k,
            "sla_snapshot": sla_snapshot,
            "timestamp":    now,
        },
        nlp_result=result.get("nlp", {}),
        rag_result=result.get("rag", {}),
        rl_decision=result.get("rl", {}),
        generated_response=result.get("response"),
        resolved=result.get("resolved", False),
        sla_breach=result.get("sla_breach", False),
        final_result={
            "strategy":       result["rl"]["strategy"],
            "action":         result["rl"]["action"],
            "top_similarity": result["rag"]["top_similarity"],
        },
    )
    doc["sla_result"] = sla_snapshot
    await db.tickets.insert_one(doc)

    return {**result, "ticket_id": doc["_id"]}


@router.get("/history/me")
async def get_history(user_id: str = Depends(get_current_user)):
    """Return all tickets for the authenticated user, newest first."""
    db = get_db()
    cursor = db.tickets.find({"user_id": user_id}).sort("timestamp", -1)

    tickets = []
    async for doc in cursor:
        tickets.append({
            "ticket_id":  doc.get("_id"),
            "raw_text":   doc.get("raw_text", "")[:100],
            "timestamp":  str(doc.get("timestamp")),
            "strategy":   doc.get("rl_decision", {}).get("strategy"),
            "resolved":   doc.get("resolved", False),
            "sla_breach": doc.get("sla_breach", False),
            "turn_count": len(doc.get("messages", [])) // 2,
        })
    return tickets


@router.get("/ticket/{ticket_id}")
async def get_ticket(ticket_id: str, user_id: str = Depends(get_current_user)):
    """Retrieve the full stored document for a single ticket."""
    db = get_db()
    doc = await db.tickets.find_one({"_id": ticket_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Ticket not found")

    doc["_id"] = str(doc["_id"])
    for msg in doc.get("messages", []):
        if hasattr(msg.get("timestamp"), "isoformat"):
            msg["timestamp"] = msg["timestamp"].isoformat()
    return doc


@router.delete("/ticket/{ticket_id}")
async def delete_ticket(ticket_id: str, user_id: str = Depends(get_current_user)):
    """Delete a ticket. Only the ticket's owner may delete it."""
    db = get_db()
    doc = await db.tickets.find_one({"_id": ticket_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if doc.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this ticket")
    await db.tickets.delete_one({"_id": ticket_id})
    return {"deleted": True}