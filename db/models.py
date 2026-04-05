"""
Plain-dict document builders — no ORM, no schemas enforced here.
All _id values are UUID strings for easy JSON serialisation.
"""
from datetime import datetime
from uuid import uuid4


def build_user_doc(username: str, email: str, hashed_password: str) -> dict:
    return {
        "_id": str(uuid4()),
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
    }


def build_ticket_doc(
    user_id: str,
    raw_text: str,
    nlp_result: dict,
    rag_result: dict,
    rl_decision: dict,
    generated_response,
    resolved: bool,
    sla_breach: bool,
) -> dict:
    """Legacy single-turn document builder (kept for backward compatibility)."""
    return {
        "_id": str(uuid4()),
        "user_id": user_id,
        "raw_text": raw_text,
        "nlp_result": nlp_result,
        "rag_result": rag_result,
        "rl_decision": rl_decision,
        "generated_response": generated_response,
        "resolved": resolved,
        "sla_breach": sla_breach,
        "timestamp": datetime.utcnow(),
    }


def build_conversation_doc(
    user_id: str,
    raw_text: str,
    first_user_msg: dict,
    first_assistant_msg: dict,
    nlp_result: dict,
    rag_result: dict,
    rl_decision: dict,
    generated_response,
    resolved: bool,
    sla_breach: bool,
    final_result: dict,
) -> dict:
    """
    Multi-turn conversation document.
    Stores the full messages array alongside the structured ML results
    so both the conversation history view and the analysis sidebar work.
    """
    return {
        "_id": str(uuid4()),
        "user_id": user_id,
        "raw_text": raw_text,
        "messages": [first_user_msg, first_assistant_msg],
        "nlp_result": nlp_result,
        "rag_result": rag_result,
        "rl_decision": rl_decision,
        "generated_response": generated_response,
        "resolved": resolved,
        "sla_breach": sla_breach,
        "final_result": final_result,
        "timestamp": datetime.utcnow(),
    }
