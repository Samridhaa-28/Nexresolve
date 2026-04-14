"""
Global model singletons + ML pipeline orchestration.

All models are loaded ONCE via load_all_models(), called from the FastAPI
lifespan hook.  Route handlers MUST NOT call load_all_models() themselves.

Override execution order (run_pipeline):
  1. NLP → state vector, intent, frustration_level
  2. RAG → top-k results, top similarity
  3. ESCALATE override if frustration_level > 0.70  (highest priority)
  4. INTENT override   if intent in [billing, duplicate, ...]
  5. RL inference      only if no override was applied
  6. SUGGEST fallback  if strategy == suggest AND top_sim < 0.50 → route
  7. conversation_end  flag
  8. Generator         only if final strategy == suggest
  9. Serialize and return
"""
from __future__ import annotations

import os
import re
import sys
import pickle
import torch
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_SCALER_PATH = str(_ROOT / "models" / "rl" / "state_scaler.pkl")
_L1_PATH     = str(_ROOT / "models" / "rl" / "l1_best.pth")
_L2_DIR      = str(_ROOT / "models" / "rl" / "best")
_INDEX_PATH  = str(_ROOT / "data"   / "retrieval" / "kb_index")

from nlp.nlp_pipeline import run_nlp_pipeline, run_rag_for_ticket as _run_rag_signals
from rl.state_builder import build_single_state
from rl.action_masking import get_action_mask
from rl.action_space import INDEX_TO_ACTION, ACTION_TO_INDEX, Strategy
from rl.level1_dqn import Level1Agent
from rl.level2_dqn import Level2Agent

_scaler    = None
_l1_agent  = None
_l2_agent  = None
_generator = None
_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SLA constants
_SLA_LIMIT_HOURS        = 24.0
_SLA_INITIAL_NORM       = 0.8        # 19.2h remaining on turn 1
_SLA_DECREASE_PER_TURN  = 0.04       # each clarification turn costs ~1h (0.04 * 24 = 0.96h)


def _clean_rag_text(text: str) -> str:
    """Strip forum noise from RAG results before passing to generator."""
    text = re.sub(r'^(hi|hello|hey|thanks|thank you|apologies for the delay)[^\n]*\n?', '', text, flags=re.IGNORECASE)
    noise_patterns = [
        r'\bi tried\b', r'\bi can confirm\b', r'\bi think\b',
        r'\bi used to\b', r'\bfor me\b', r'\bin my case\b',
        r'\bworks for me\b', r'\bsame issue here\b',
        r'\bthanks for (the|your) report\b',
        r'\bapologies for the delay\b',
        r'\bencountered this exact issue\b',
        r'\bthe issue seems\b',
        r'\bindeed related to\b',
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[url\]|\[SEP\]|\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def pick_best_rag_result(rag_results: list) -> str | None:
    VAGUE_PHRASES = [
        "resolved now", "this is resolved", "fixed in", "closing this",
        "duplicate of", "same issue", "no code change", "works for me",
        "cannot reproduce", "please reopen", "marked as", "see #",
    ]
    MIN_WORDS = 15
    
    ACTION_WORDS = [
    "reduce", "disable", "restart", "clear",
    "update", "install", "reinstall", "set"
    ]

    for result in rag_results:
        solution = result["solution"].strip().lower()
        word_count = len(solution.split())
        if not any(phrase in solution for phrase in VAGUE_PHRASES) and word_count >= MIN_WORDS:
            return result["solution"]
        if any(word in solution for word in ACTION_WORDS):
            return result["solution"]

    return rag_results[0]["solution"] if rag_results else None


'''def clean_generated_output(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen: set[str] = set()
    cleaned: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent.split()) < 4:
            continue
        normalized = sent.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(sent)
    return " ".join(cleaned) if cleaned else text.strip()'''

def clean_generated_output(text: str) -> str:
    if not text:
        return text

    text = text.strip()

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    seen = set()
    steps = []

    for sent in sentences:
        sent = sent.strip()

        # Skip very short lines
        if len(sent.split()) < 4:
            continue

        key = sent.lower()

        # Remove duplicates
        if key in seen:
            continue
        seen.add(key)

        # Remove conversational junk
        if any(x in key for x in [
            "i encountered", "for me", "in my case",
            "thanks", "apologies", "the issue seems",
            "i think", "i believe"
        ]):
            continue

        steps.append(sent)

    if not steps:
        return text

    # 🔥 Detect if actionable
    action_words = [
        "reduce", "restart", "check", "disable",
        "run", "clear", "install", "update", "verify", "set"
    ]

    is_actionable = any(
        any(word in step.lower() for word in action_words)
        for step in steps
    )

    # Format output
    formatted = []
    for i, step in enumerate(steps[:5], 1):
        if is_actionable:
            formatted.append(f"{i}. {step}")   # steps
        else:
            formatted.append(f"• {step}")      # bullets

    return "\n".join(formatted)

_CLARIFY_QUESTION_MAP = {
    "ask_version":      "What version of the software or package are you using?",
    "ask_platform":     "What operating system or platform are you running on?",
    "ask_error_type":   "Could you share the exact error message you are seeing?",
    "ask_hardware":     "What hardware (GPU/CPU model) are you using?",
    "ask_steps":        "Could you describe the exact steps that led to this issue?",
    "ask_uncertainty":  "Could you clarify exactly what is happening? Any additional context helps.",
    "ask_vague_request": "Could you describe your issue in more detail so we can assist you better?",
}


def load_all_models() -> None:
    global _scaler, _l1_agent, _l2_agent, _generator, _device

    with open(_SCALER_PATH, "rb") as fh:
        _scaler = pickle.load(fh)
    print("[startup] State scaler loaded")

    _l1_agent = Level1Agent(state_dim=37)
    ckpt = torch.load(_L1_PATH, map_location=_device)
    _l1_agent.online_net.load_state_dict(ckpt["online_state_dict"])
    _l1_agent.online_net.eval()
    print(f"[startup] L1 agent loaded on {_device}")

    _l2_agent = Level2Agent(state_dim=37)
    _l2_agent.load_all(_L2_DIR)
    for net in _l2_agent.online_nets.values():
        net.eval()
    print("[startup] L2 agent loaded")

    from generation.generator import ResponseGenerator
    try:
        _generator = ResponseGenerator(model_type="FLAN")
        print(f"[startup] Generator loaded on {_generator.device}")
    except RuntimeError as exc:
        if _is_oom(exc):
            print(f"[startup] GPU OOM loading generator — retrying on CPU: {exc}")
            _generator = ResponseGenerator(model_type="FLAN", device="cpu")
            print("[startup] Generator loaded on CPU (fallback)")
        else:
            raise


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg


def _to_python(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_python(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_to_python(i) for i in v]
    return v


def run_pipeline(ticket_text: str, user_id: str, turn_count: int = 1) -> dict:
    """
    End-to-end inference for a single ticket.

    Args:
        ticket_text: The (possibly concatenated) ticket text.
        user_id:     Authenticated user ID.
        turn_count:  How many turns have occurred (1 = first turn).
                     SLA remaining decreases by _SLA_DECREASE_PER_TURN each turn.
    """
    # ── 1. NLP ────────────────────────────────────────────────────────────────
    nlp_result: dict = run_nlp_pipeline(ticket_text)

    # ── 2. RAG (single retrieval — reused for state signals and solution) ──────
    _rag_signal = _run_rag_signals(
        clean_text        = ticket_text,
        intent_label      = nlp_result.get("intent_group"),
        intent_confidence = float(nlp_result.get("confidence_score", 1.0)),
        top_k             = 5,
        index_path        = _INDEX_PATH,
        return_full       = True,
    )
    retrieved: list = _rag_signal.get("retrieved_optimised") or _rag_signal.get("retrieved_raw") or []
    top_sim: float  = float(retrieved[0]["similarity_score"]) if retrieved else 0.0

    rag_results = []
    rag_top_k   = []
    for item in retrieved:
        sol = item.get("solution_comments", "") or ""
        rag_results.append({
            "solution": sol,
            "score":    round(float(item.get("similarity_score", 0.0)), 4),
        })
        rag_top_k.append({
            "solution": sol[:200],
            "score":    round(float(item.get("similarity_score", 0.0)), 4),
        })

    top_solution = pick_best_rag_result(rag_results)

    # ── 3. State row ──────────────────────────────────────────────────────────
    # SLA remaining decreases by _SLA_DECREASE_PER_TURN for each clarification turn
    sla_remaining_norm  = max(0.0, round(_SLA_INITIAL_NORM - _SLA_DECREASE_PER_TURN * (turn_count - 1), 4))
    sla_limit_hours     = _SLA_LIMIT_HOURS
    sla_remaining_hours = round(sla_limit_hours * sla_remaining_norm, 1)
    sla_breach_flag     = 1 if sla_remaining_hours <= 0 else 0

    row = {
        **nlp_result,
        "clean_text":           ticket_text,
        "text_length":          len(ticket_text),
        "word_count":           len(ticket_text.split()),
        "question_mark_flag":   int("?" in ticket_text),
        "has_solution_comment": 0,
        "turn_count":           turn_count,
        "sla_breach_flag":      sla_breach_flag,
        "sla_remaining_norm":   sla_remaining_norm,
        "sla_limit_hours":      sla_limit_hours,
        "interaction_depth":    max(0, turn_count - 1),
        "reassignment_count":   0,
        "reopen_count":         0,
        "resolution_success":   0,
        "confidence_band":      nlp_result.get("confidence_band", "medium"),
        "uncertainty_flag":     nlp_result.get("uncertainty_flag", 0),
        "rl_recommendation":    "route_directly" if nlp_result.get("intent_group") in {
                                    "billing", "general", "feature_request", "duplicate"
                                } else "clarify_first",
        "top_tier":             _rag_signal.get("top_tier", "tier3_minimal"),
        "max_sim":              _rag_signal.get("max_sim", 0.0),
        "avg_sim":              _rag_signal.get("avg_sim", 0.0),
        "sim_spread":           _rag_signal.get("sim_spread", 0.0),
        "knowledge_gap_flag":   _rag_signal.get("knowledge_gap_flag", 1),
    }

    state: np.ndarray      = build_single_state(row, index_path=_INDEX_PATH)
    state_norm: np.ndarray = _scaler.transform([state])[0]
    mask: np.ndarray       = get_action_mask(state_norm)

    frustration_level = float(nlp_result.get("frustration_level", 0.0))
    intent            = (nlp_result.get("intent_group") or "other").lower()
    urgency_score     = float(nlp_result.get("urgency_score", 0.0))
    urgent_flag       = int(nlp_result.get("urgent_flag", 0))

    override_applied       = False
    strategy               = None
    action_name            = None
    strategy_idx           = 0
    action_idx             = 0
    response               = None
    generator_output       = None
    clarification_question = None
    conversation_end       = False

    # Step A — SLA breach override
    if sla_breach_flag == 1:
        strategy      = "escalate"
        action_name   = "escalate_human"
        strategy_idx  = Strategy.ESCALATE.value
        action_idx    = ACTION_TO_INDEX["escalate_human"]
        response      = (
            "The SLA deadline for this ticket has been breached. "
            "Your case has been immediately escalated to a human specialist "
            "for urgent resolution."
        )
        conversation_end = True
        override_applied = True

    # Step B — ESCALATE override (high frustration)
    elif frustration_level > 0.70:
        strategy      = "escalate"
        action_name   = "escalate_human"
        strategy_idx  = Strategy.ESCALATE.value
        action_idx    = ACTION_TO_INDEX["escalate_human"]
        response      = (
            "We understand this issue has been frustrating. "
            "Your ticket has been escalated to a human specialist "
            "who will contact you shortly."
        )
        conversation_end = True
        override_applied = True

    # Step C — INTENT override
    elif intent in ("billing", "billing_issue", "account_issue"):
        strategy      = "route"
        action_name   = "route_billing"
        strategy_idx  = Strategy.ROUTE.value
        action_idx    = ACTION_TO_INDEX["route_billing"]
        response      = (
            "Your billing issue has been routed to our billing and accounts team. "
            "They will review your case and contact you directly."
        )
        conversation_end = True
        override_applied = True

    elif intent == "duplicate":
        strategy      = "route"
        action_name   = "route_duplicate"
        strategy_idx  = Strategy.ROUTE.value
        action_idx    = 0  # no exact action for duplicate; maps to route_bug
        response      = (
            "This appears to be a known issue that is already being tracked. "
            "Your ticket has been linked to the existing report."
        )
        conversation_end = True
        override_applied = True

    # Step D — RL inference
    if not override_applied:
        strategy_idx = int(_l1_agent.select_action(state_norm, mask, greedy=True))
        strategy     = Strategy(strategy_idx).name.lower()
        action_idx   = int(_l2_agent.select_action(state_norm, strategy_idx, mask))
        action_name  = INDEX_TO_ACTION[action_idx]

        # Step E — SUGGEST fallback: low similarity → route
        if strategy == "suggest" and top_sim < 0.50:
            strategy     = "route"
            action_name  = "route_technical"
            response     = (
                "We couldn't find a confident match for your issue. "
                "Your ticket has been routed to our support team for review."
            )
            conversation_end = True

    # Step F — conversation_end
    if not conversation_end:
        conversation_end = strategy in ("suggest", "route", "escalate")

    # Step G — Generator
    if strategy == "suggest" and top_sim >= 0.50:
        if not top_solution:
            strategy    = "route"
            action_name = "route_technical"
            response    = (
                "We could not find a relevant solution for your issue. "
                "Your ticket has been routed to our support team for manual review."
            )
            conversation_end = True
        else:
            entities = {
                "version":    int(nlp_result.get("has_version",    0)),
                "error_type": int(nlp_result.get("has_error_type", 0)),
                "platform":   int(nlp_result.get("has_platform",   0)),
                "hardware":   int(nlp_result.get("has_hardware",   0)),
            }

            cleaned_solution = _clean_rag_text( top_solution)

            raw_output = _safe_generate(
                ticket_summary=ticket_text[:300],
                intent=nlp_result.get("intent_group", "other"),
                entities=entities,
                retrieved_solution=cleaned_solution,
            )
            generator_output = clean_generated_output(raw_output or "")
            response = generator_output if generator_output else (
                "We found a potential solution. Please contact support if the issue persists."
            )

    # Step H — CLARIFY
    if strategy == "clarify":
        clarification_question = _CLARIFY_QUESTION_MAP.get(
            action_name,
            f"To better assist you, could you provide more details? "
            f"({action_name.replace('_', ' ')})",
        )
        response = clarification_question

    final_strategy = strategy

    return {
        "nlp": _to_python(nlp_result),
        "rag": {
            "top_similarity":  float(top_sim),
            "retrieved_count": int(len(retrieved)),
            "top_solution":    top_solution[:300] if top_solution else "",
            "top_k":           rag_top_k,
        },
        "rl": {
            "strategy":     final_strategy,
            "strategy_idx": int(strategy_idx),
            "action":       action_name,
            "action_idx":   int(action_idx),
        },
        "sla": {
            "sla_limit_hours":     sla_limit_hours,
            "sla_remaining_norm":  sla_remaining_norm,
            "sla_remaining_hours": sla_remaining_hours,
            "sla_breach_flag":     sla_breach_flag,
        },
        "response":               response,
        "generator_output":       generator_output,
        "clarification_question": clarification_question,
        "conversation_end":       conversation_end,
        "resolved":               bool(final_strategy == "suggest" and top_sim >= 0.50),
        "sla_breach":             bool(sla_breach_flag == 1),
        "urgency_score":          urgency_score,
        "urgent_flag":            urgent_flag,
    }


def _safe_generate(ticket_summary: str, intent: str, entities: dict,
                   retrieved_solution: str) -> str | None:
    global _generator

    def _call() -> str:
        return _generator.generate(
            ticket_summary=ticket_summary,
            intent=intent,
            entities=entities,
            retrieved_solution=retrieved_solution,
        )

    try:
        return _call()
    except RuntimeError as exc:
        if not _is_oom(exc):
            print(f"[pipeline] Generation RuntimeError (skipped): {exc}")
            return None
        print(f"[pipeline] GPU OOM during generation — switching to CPU: {exc}")
        try:
            _generator.device = "cpu"
            _generator.model  = _generator.model.cpu()
            return _call()
        except Exception as retry_exc:
            print(f"[pipeline] CPU retry also failed: {retry_exc}")
            return None
    except Exception as exc:
        print(f"[pipeline] Generation error (skipped): {exc}")
        return None