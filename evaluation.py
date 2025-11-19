# evaluation.py
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import os
import logging
import google.generativeai as genai
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logger.error("FATAL: GEMINI_API_KEY environment variable not set.")

# Import helpers
from evaluation_helpers import (
    emb, cosine_sim, load_canonical_for_role, find_best_canonical_match,
    evaluate_resume_answer_with_llm, compute_cci_from_components, load_historical_ccis, clean_text,generate_human_summaries_with_llm,evaluate_answer_with_gemini
)

# For robust logging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_interview(session_id: str,
                       Name: str,
                       transcript_log: List[Dict[str, Any]],
                       role: str,
                       malpractice_events: Optional[List[Dict[str, Any]]] = None,
                       resume_info: Optional[Dict[str, Any]] = None,
                       role_requirements: Optional[List[str]] = None,
                       num_role_canonical: int = 5) -> Dict[str, Any]:
    """
    Updated evaluate_interview that returns the extra human-readable fields
    and numeric sub-scores requested by the user.
    """
    malpractice_events = malpractice_events if malpractice_events is not None else []
    cleaned_events = []
    skipped_no_face = False
    for ev in malpractice_events:
        if not skipped_no_face and ev.get("type") == "no_face_detected":
            skipped_no_face = True
            continue
        cleaned_events.append(ev)

    malpractice_events = cleaned_events

    # Malpractice summary
    malpractice_count = len(malpractice_events)
    malpractice_types = list({e.get("type") for e in malpractice_events if isinstance(e, dict)})
    score_deduction = 0.0
    if malpractice_count > 0:
        score_deduction = min(len(malpractice_types) * 2.0 + malpractice_count * 1.0, 30.0)
    malpractice_summary = {"count": malpractice_count, "types": malpractice_types, "deduction": round(score_deduction, 2)}

    # Load canonical Q/A pairs
    canonical_list = load_canonical_for_role(role)
    logger.info("Loaded %d canonical items for role '%s'", len(canonical_list), role)

    # Build QA pairs from transcript_log (list of dicts with question/answer/type)
    qa_pairs = []
    for item in transcript_log:
        q = item.get("question") if isinstance(item, dict) else None
        a = item.get("answer") if isinstance(item, dict) else None
        t = item.get("type") if isinstance(item, dict) else None
        if q is None and isinstance(item, str):
            q = item
        if q:
            qa_pairs.append((q or "", a or "", t or "standard"))

    # Precompute canonical embeddings
    canonical_embs = []
    for c in canonical_list:
        cand_text = c.get("canonical_answer") or c.get("question") or ""
        canonical_embs.append(emb(cand_text))

    per_q_reports = []
    canonical_scores = []
    q_a_sims = []
    resume_scores = []

    # Per-question processing
    for i, (q_text, a_text, t) in enumerate(qa_pairs):
        q_emb = emb(q_text)
        a_emb = emb(a_text)

        chosen_idx = None
        canonical_answer = ""
        canonical_similarity = 0.0
        canonical_score = 0.0
        qa_sim = 0.0
        resume_eval_result = None

        qa_sim = cosine_sim(q_emb, a_emb)
        q_a_sims.append(qa_sim)
        if t == "role":
            # Try positional fallback first, then use semantic matching
            chosen_idx = None
            try:
                # positional fallback: if canonical_list looks aligned with transcript order
                if i < len(canonical_list):
                    cand_q = canonical_list[i].get("question", "")
                    # small normalization check: compare prefixes (avoid exact match requirement)
                    if cand_q and clean_text(cand_q).lower()[:60] in clean_text(q_text).lower()[:200]:
                        chosen_idx = i
                # if still None, run semantic matcher
                if chosen_idx is None and canonical_list:
                    chosen_idx = find_best_canonical_match(q_text, canonical_list)
            except Exception:
                logger.exception("Error picking canonical match for role question")

            # Prepare defaults
            canonical_answer = ""
            canonical_similarity = 0.0
            canonical_score = 0.0
            gem_score = None
            gem_reason = ""

            # If we have a canonical match -> compute embedding-based similarity & score
            if chosen_idx is not None and chosen_idx < len(canonical_list):
                canonical_answer = canonical_list[chosen_idx].get("canonical_answer", "") or ""
                try:
                    cand_emb = canonical_embs[chosen_idx] if chosen_idx < len(canonical_embs) else emb(canonical_answer)
                    canonical_similarity = cosine_sim(a_emb, cand_emb)
                    canonical_score = round(max(0.0, min(1.0, canonical_similarity)) * 100.0, 2)
                except Exception:
                    logger.exception("Error computing canonical similarity/score")
                    canonical_similarity = 0.0
                    canonical_score = 0.0

            # Optionally grade with Gemini (if you want it even when no canonical match)
            try:
                # evaluate_answer_with_gemini should gracefully accept empty canonical_answer if none
                gem_res = evaluate_answer_with_gemini(q_text, a_text, canonical_answer)
                gem_score = gem_res.get("score")
                gem_reason = gem_res.get("reason", "")
            except Exception as e:
                logger.exception("Gemini grading exception: %s", e)
                gem_score = None
                gem_reason = f"Gemini error: {e}"

            # Combine SBERT canonical_score with Gemini score (weights configurable)
            EMBEDDING_WEIGHT = 0.5
            GEMINI_WEIGHT = 0.5
            if gem_score is not None:
                try:
                    combined_score = round((EMBEDDING_WEIGHT * canonical_score) + (GEMINI_WEIGHT * float(gem_score)), 2)
                except Exception:
                    combined_score = canonical_score
            else:
                combined_score = canonical_score

            # Append result to per-question reports (always append for role questions)
            per_q_reports.append({
                "question": q_text,
                "answer": a_text,
                "type": "role",
                "canonical_matched_index": chosen_idx,
                "canonical_answer": canonical_answer,
                "canonical_similarity": round(canonical_similarity, 3),
                "canonical_score": canonical_score,
                "gemini_score": gem_score,
                "gemini_reason": gem_reason,
                "combined_role_score": combined_score,
                "q_a_similarity": round(qa_sim, 3),
                "answer_length": len(a_text.split()) if a_text else 0,
                "resume_eval": resume_eval_result
            })

            # Use combined_score (if numeric) for aggregate canonical scoring
            try:
                canonical_scores.append(float(combined_score))
            except Exception:
                # if combined_score isn't numeric, ignore
                pass

            # then continue to next question (if your loop relies on continue)
            continue

        elif t in ("resume", "follow-up"):
            try:
                resume_eval_result = evaluate_resume_answer_with_llm(resume_info or {}, q_text, a_text, role=role)
                if resume_eval_result.get("resume_score") is not None:
                    resume_scores.append(float(resume_eval_result["resume_score"]))
            except Exception:
                logger.exception("LLM resume eval failed")
                resume_eval_result = {"resume_score": None, "evidence": [], "weaknesses": [], "notes": "LLM error"}

            per_q_reports.append({
            "question": q_text,
            "answer": a_text,
            "type": t,
            "resume_eval": resume_eval_result
        })

    # Components aggregation (existing logic)
    canonical_component_raw = float(np.mean(canonical_scores)) if canonical_scores else 0.0
    canonical_component = (canonical_component_raw / 100.0) * 30.0

    resume_component_raw = float(np.mean(resume_scores)) if resume_scores else None
    resume_component = (resume_component_raw / 100.0) * 30.0 if resume_component_raw is not None else None

    completeness_raw = float(np.mean(q_a_sims)) if q_a_sims else 0.0
    completeness_component = completeness_raw * 25.0

    # conciseness (heuristic)
    lens = []
    for _, a, _ in qa_pairs:
        tokens = len((a or "").split())
        if tokens == 0:
            lens.append(0.0)
        elif tokens < 8:
            lens.append(0.5)
        elif tokens > 200:
            lens.append(0.6)
        else:
            lens.append(1.0)
    conciseness_component = float(np.mean(lens)) * 15.0 if lens else 0.0

    # fluency proxy
    fluencies = []
    for _, a, _ in qa_pairs:
        if not a or len(a.split()) < 2:
            fluencies.append(0.3)
            continue
        sentences = re.split(r'[.!?]+', a)
        sent_lens = [len(s.split()) for s in sentences if s.strip()]
        if not sent_lens:
            fluencies.append(0.5)
            continue
        avg = sum(sent_lens) / len(sent_lens)
        if avg < 8:
            fluencies.append(0.7)
        elif avg > 30:
            fluencies.append(0.6)
        else:
            fluencies.append(1.0)
    fluency_component = float(np.mean(fluencies)) * 15.0 if fluencies else 0.0

    if resume_component is not None:
        role_relevance_component = round((0.6 * canonical_component) + (0.4 * resume_component), 2)
    else:
        role_relevance_component = round(canonical_component, 2)

    components = {
        "role_relevance": role_relevance_component,  # 0-30
        "canonical_component": round(canonical_component, 2),
        "resume_component": round(resume_component, 2) if resume_component is not None else None,
        "completeness": round(completeness_component, 2),
        "conciseness": round(conciseness_component, 2),
        "fluency": round(fluency_component, 2),
    }

    # Map components to normalized 0-100 for CCI computation
    norm_components = {
        "role_relevance": (components["role_relevance"] / 30.0) * 100.0,
        "completeness": (components["completeness"] / 25.0) * 100.0,
        "conciseness": (components["conciseness"] / 15.0) * 100.0,
        "fluency": (components["fluency"] / 15.0) * 100.0
    }

    # compute CCI (existing helper)
    historical = load_historical_ccis(role)
    cci_info = compute_cci_from_components(norm_components, weights=None, historical_scores=historical, calibrate=True)
    final_raw_score = round(cci_info["cci_raw"], 1)

    # apply malpractice deduction
    final_score_after_deduction = max(0, final_raw_score - malpractice_summary["deduction"])

    # ----------------------------
    # NEW: compute human-friendly numeric subscores (0-10)
    # ----------------------------
    try:
        conceptual_correctness = round(min(10.0, (components["canonical_component"] / 30.0) * 10.0), 2)
    except Exception:
        conceptual_correctness = 0.0

    try:
        reasoning_depth = round(min(10.0, (components["completeness"] / 25.0) * 10.0), 2)
    except Exception:
        reasoning_depth = 0.0

    try:
        precision_grounding = round(min(10.0, (components["role_relevance"] / 30.0) * 10.0), 2)
    except Exception:
        precision_grounding = 0.0

    try:
        communication_clarity = round(min(10.0, (components["fluency"] / 15.0) * 10.0), 2)
    except Exception:
        communication_clarity = 0.0

    # creativity/insight: blend of completeness and conciseness (both reflect novelty & succinctness)
    try:
        creativity_insight = round(min(10.0, ((components["completeness"] / 25.0) * 5.0) + ((components["conciseness"] / 15.0) * 5.0)), 2)
    except Exception:
        creativity_insight = 0.0

    numeric_subscores = {
        "Conceptual correctness": conceptual_correctness,
        "Reasoning depth": reasoning_depth,
        "Precision grounding": precision_grounding,
        "Communication clarity": communication_clarity,
        "Creativity / insight": creativity_insight
    }

    # Use LLM (with fallback) to get human summaries
    try:
        llm_out = generate_human_summaries_with_llm(
            session_id=session_id,
            candidate_name=Name or "",
            role=role or "",
            numeric_subscores=numeric_subscores,
            overall_cci=cci_info.get("cci_raw", 0.0),
            malpractice_summary=malpractice_summary
        )
    except Exception:
        logger.exception("LLM human summary generation failure, using fallback")
        llm_out = generate_human_summaries_with_llm(
            session_id=session_id,
            candidate_name=Name or "",
            role=role or "",
            numeric_subscores=numeric_subscores,
            overall_cci=cci_info.get("cci_raw", 0.0),
            malpractice_summary=malpractice_summary
        )

    trajectory = llm_out.get("trajectory")
    strength = llm_out.get("strength")
    risk = llm_out.get("risk")
    recommendation = llm_out.get("recommendation")
    badge = llm_out.get("Badge")

    # Build final report
    report = {
        "session_id": session_id,
        "Name": Name,
        "role": role,
        "start_time": transcript_log[0].get("timestamp") if transcript_log and isinstance(transcript_log[0], dict) else datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),

        # Scores (existing)
        "overall_score": round(final_score_after_deduction, 1),
        "score_before_malpractice": final_raw_score,
        "cci": {"raw": cci_info["cci_raw"], "percentile_vs_role": cci_info.get("cci_percentile")},
        "components": components,

        # New numeric readable subs (0-10)
        "subscores": numeric_subscores,

        # LLM / human readable summaries
        "trajectory": trajectory,
        "strength": strength,
        "risk": risk,
        "recommendation": recommendation,
        "Badge": badge,

        # Malpractice and per-question
        "malpractice_detected": malpractice_summary,
        "per_question": per_q_reports,
        "resume_info_evaluated": resume_info or {},
         
    }

    return report