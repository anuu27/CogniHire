# evaluation_helpers.py
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import math
import statistics
import time
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import logging
import google.generativeai as genai
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Initialize SBERT model once (costly)
_SBERT = None
def _get_sbert():
    global _SBERT
    if _SBERT is None:
        _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SBERT

# Gemeni model name (you can override by passing gemini_model to functions)
_GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# ------------------------
# Text / embedding helpers
# ------------------------
def clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())

def emb(text: str) -> np.ndarray:
    txt = clean_text(text)
    try:
        model = _get_sbert()
        return model.encode(txt, convert_to_numpy=True)
    except Exception:
        # safe fallback zero vector of model dim if model not loaded
        try:
            dim = _get_sbert().get_sentence_embedding_dimension()
            return np.zeros((dim,), dtype=float)
        except Exception:
            return np.zeros((384,), dtype=float)

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    try:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a / na, b / nb))
    except Exception:
        return 0.0

# ------------------------
# Canonical loader & matcher
# ------------------------
def load_canonical_for_role(role: str) -> List[Dict[str, str]]:
    """
    Loads canonical Q/A pairs from data/role_questions/<role>.json
    Expected file structure (as in your screenshot):
    {
      "role": "AI Engineer",
      "record_count": ...,
      "desired_questions": ...,
      "questions": [
         {"question": "...", "canonical_answer": "...", "rubric_points": [...]},
         ...
      ]
    }
    Returns list of dicts: [{"question": "...", "canonical_answer": "..."}, ...]
    """
    key = re.sub(r'[^a-z0-9]+', '_', role.lower()).strip('_')
    file_path = Path("data/role_questions") / f"{key}.json"

    if not file_path.exists():
        # try alternative names
        alt = Path("data/role_questions") / f"{key}_canonical.json"
        if alt.exists():
            file_path = alt
        else:
            return []

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    cleaned = []
    if isinstance(raw, dict):
        qlist = raw.get("questions") or raw.get("question_list") or raw.get("items")
        if isinstance(qlist, list):
            for it in qlist:
                if isinstance(it, str):
                    cleaned.append({"question": it, "canonical_answer": ""})
                elif isinstance(it, dict):
                    q = it.get("question") or it.get("q") or ""
                    ca = it.get("canonical_answer") or it.get("canonical") or it.get("answer") or ""
                    cleaned.append({"question": clean_text(q), "canonical_answer": clean_text(ca)})
        else:
            # fallback: if dict contains question->canonical mapping
            # collect string values
            for v in raw.values():
                if isinstance(v, str) and v.strip():
                    cleaned.append({"question": v.strip(), "canonical_answer": ""})
    elif isinstance(raw, list):
        for it in raw:
            if isinstance(it, str):
                cleaned.append({"question": it, "canonical_answer": ""})
            elif isinstance(it, dict):
                q = it.get("question") or it.get("q") or ""
                ca = it.get("canonical_answer") or it.get("canonical") or it.get("answer") or ""
                cleaned.append({"question": clean_text(q), "canonical_answer": clean_text(ca)})
    return cleaned

def find_best_canonical_match(question_text: str, canonical_list: List[Dict[str, str]]) -> Optional[int]:
    """Return index of best matching canonical question by SBERT similarity (question->canonical_question)."""
    if not canonical_list:
        return None
    q_emb = emb(question_text)
    best_idx = None
    best_sim = -1.0
    for i, item in enumerate(canonical_list):
        cand_q = item.get("question", "") or ""
        s = cosine_sim(q_emb, emb(cand_q))
        if s > best_sim:
            best_sim = s
            best_idx = i
    return best_idx

# ------------------------
# LLM resume-answer evaluator (per resume QA)
# ------------------------
def _safe_extract_json(text: str):
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception:
        return None
def evaluate_resume_answer_with_llm(resume_info: Dict[str, Any],
                                    question: str,
                                    answer: str,
                                    role: str = "",
                                    gemini_model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Robust helper to call Gemini to evaluate a resume-focused QA pair.
    Works with SDKs that don't accept extra kwargs like max_output_chars.
    Returns a dict with keys: resume_score, evidence, weaknesses, notes, raw_llm_text.
    """
    model = gemini_model or genai.GenerativeModel(_GEMINI_MODEL_NAME)

    prompt = f"""
You are an expert technical interviewer and resume verifier.

Given the candidate's structured resume JSON, the role, a specific resume-focused interview question, and the candidate's answer,
return ONLY valid JSON with the following keys:
- resume_score (0-100): numeric estimate of how well the answer demonstrates resume claims & role-fit.
- evidence: array (0-3) of short strings quoting or summarizing parts of the answer that support the score.
- weaknesses: array (0-3) of missing details or red flags.
- notes: one short paragraph (1-2 sentences) explaining the score.

Resume JSON: {json.dumps(resume_info, ensure_ascii=False)}
Role: "{role}"
Question: "{question}"
Candidate Answer: "{answer}"

Example JSON:
{{ "resume_score": 78, "evidence": ["used Docker & Kubernetes"], "weaknesses": ["no monitoring mentioned"], "notes": "Concrete deployment steps but missing monitoring." }}
"""

    raw_text = ""
    try:
        # primary attempt: basic generate_content(prompt)
        resp = model.generate_content(prompt)
        raw_text = getattr(resp, "text", None) or \
                   (resp[0].content if isinstance(resp, (list, tuple)) and hasattr(resp[0], "content") else str(resp))
    except TypeError as te:
        # retry a plain call if signature mismatch (some SDKs vary)
        try:
            resp = model.generate_content(prompt)
            raw_text = getattr(resp, "text", None) or \
                       (resp[0].content if isinstance(resp, (list, tuple)) and hasattr(resp[0], "content") else str(resp))
        except Exception as e:
            logger.exception("LLM evaluate_retry failed")
            return {"resume_score": None, "evidence": [], "weaknesses": [], "notes": f"LLM error: {e}", "raw_llm_text": raw_text}
    except Exception as e:
        logger.exception("LLM evaluate failed")
        return {"resume_score": None, "evidence": [], "weaknesses": [], "notes": f"LLM error: {e}", "raw_llm_text": raw_text}

    # parse JSON out of LLM text
    parsed = _safe_extract_json(raw_text or "")
    if not parsed:
        # helpful debug in notes: include raw LLM output so you can inspect it in the report
        return {"resume_score": None, "evidence": [], "weaknesses": [], "notes": (raw_text or "No JSON returned"), "raw_llm_text": raw_text}

    # sanitize score
    score = parsed.get("resume_score")
    try:
        if score is not None:
            score = float(score)
            score = max(0.0, min(100.0, score))
    except Exception:
        score = None

    return {
        "resume_score": score,
        "evidence": parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else [],
        "weaknesses": parsed.get("weaknesses") if isinstance(parsed.get("weaknesses"), list) else [],
        "notes": parsed.get("notes") or "",
        "raw_llm_text": raw_text
    }


# ------------------------
# CCI helpers
# ------------------------
def z_to_percentile(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0

def compute_cci_from_components(components: Dict[str, float],
                                weights: Optional[Dict[str, float]] = None,
                                historical_scores: Optional[List[float]] = None,
                                calibrate: bool = True) -> Dict[str, Optional[float]]:
    """
    components: dict of component scores already on 0-100 scale
    weights: dict with same keys -> weight values (sums need not equal 1)
    historical_scores: optional list of previous CCI raw values (for percentile)
    Returns: {"cci_raw": float, "cci_percentile": float or None}
    """
    default_weights = {
        "role_relevance": 35.0,
        "completeness": 30.0,
        "conciseness": 15.0,
        "fluency": 20.0
    }
    w = weights or default_weights
    used_keys = [k for k in components.keys() if k in w]
    if not used_keys:
        vals = [float(v) for v in components.values()] if components else []
        cci_raw = float(sum(vals) / len(vals)) if vals else 0.0
        return {"cci_raw": round(cci_raw, 2), "cci_percentile": None}

    numerator = 0.0
    denom = 0.0
    for k in used_keys:
        comp_val = max(0.0, min(100.0, float(components[k])))
        weight = float(w[k])
        numerator += comp_val * weight
        denom += weight
    cci_raw = (numerator / denom) if denom else 0.0
    cci_raw = max(0.0, min(100.0, cci_raw))

    cci_percentile = None
    if calibrate and historical_scores:
        try:
            mean = statistics.mean(historical_scores)
            stdev = statistics.pstdev(historical_scores) if len(historical_scores) > 1 else 0.0
            if stdev > 0:
                z = (cci_raw - mean) / stdev
                cci_percentile = z_to_percentile(z)
            else:
                cci_percentile = 50.0 if abs(cci_raw - mean) < 1e-6 else (100.0 if cci_raw > mean else 0.0)
        except Exception:
            cci_percentile = None

    return {"cci_raw": round(cci_raw, 2), "cci_percentile": round(cci_percentile, 2) if cci_percentile is not None else None}

def load_historical_ccis(role: str, base_dir: Path = Path("data/interviews")) -> List[float]:
    scores = []
    try:
        for rp in base_dir.glob("*_report.json"):
            try:
                d = json.loads(rp.read_text(encoding="utf-8"))
                if d.get("role") == role:
                    v = d.get("cci", {}) or {}
                    s = v.get("raw") or d.get("overall_score")
                    if s is not None:
                        scores.append(float(s))
            except Exception:
                continue
    except Exception:
        pass
    return scores
# Put this near evaluate_resume_answer_with_llm or in evaluation_helpers.py

def generate_human_summaries_with_llm(session_id: str,
                                      candidate_name: str,
                                      role: str,
                                      numeric_subscores: dict,
                                      overall_cci: float,
                                      malpractice_summary: dict,
                                      gemini_model=None,
                                      max_retries: int = 2) -> dict:
    """
    Ask LLM to produce short human-readable fields:
      - trajectory
      - strength
      - risk
      - recommendation

    Returns dict with those keys. If LLM fails, returns simple rule-based fallback.
    """
    model = gemini_model or genai.GenerativeModel(_GEMINI_MODEL_NAME)

    prompt = f"""
You are an expert recruiting evaluator. Given numeric sub-scores and basic context, return a JSON object with keys:
- trajectory: one short sentence describing candidate's expected career/skill trajectory.
- strength: one short sentence naming the top strength and its score (0-10).
- risk: one short sentence describing major risk(s) (malpractice, low subs).
- recommendation: one short sentence recommendation for next step (hire/onsite/take-home/reject).
-Badge: one of "Hire Ready", "Promising with some Gaps", "Not Recommended" based on overall CCI and malpractice.
Session: "{session_id}"
Candidate: "{candidate_name}"
Role: "{role}"
Overall CCI (0-100): {overall_cci}
Malpractice summary: {json.dumps(malpractice_summary)}
Numeric subscores (0-10): {json.dumps(numeric_subscores)}

Return ONLY valid JSON, example:
{{ "trajectory": "...", "strength": "...", "risk": "...", "recommendation": "...", "Badge": "..." }}
"""
    attempt = 0
    raw_text = ""
    while attempt <= max_retries:
        try:
            resp = model.generate_content(prompt)
            # resp handling is a bit SDK-specific; try common attributes
            raw_text = getattr(resp, "text", None) or (resp[0].content if isinstance(resp, (list,tuple)) and hasattr(resp[0],"content") else str(resp))
            parsed = None
            try:
                # try extracting JSON object from output
                m = re.search(r'\{[\s\S]*\}', raw_text)
                if m:
                    parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

            if parsed and all(k in parsed for k in ("trajectory","strength","risk","recommendation")):
                # sanitize strings
                return {k: str(parsed[k]).strip() for k in ("trajectory","strength","risk","recommendation")}
        except Exception as e:
            # log and retry
            logger.exception("LLM summary generation failed (attempt %d): %s", attempt, e)
        attempt += 1
        time.sleep(0.5*(attempt))  # small backoff

    # ----- Fallback rule-based summaries (guaranteed) -----
    # pick best subscore
    best_name, best_val = max(numeric_subscores.items(), key=lambda x: x[1])
    trajectory = ("Strong upward trajectory — candidate demonstrates high competency."
                  if overall_cci >= 80 else
                  "Solid trajectory — good competency with room to grow." if overall_cci >= 60 else
                  "Mixed trajectory — some strengths but notable gaps." if overall_cci >= 40 else
                  "Low trajectory — candidate shows limited role readiness.")

    if malpractice_summary.get("count", 0) > 0:
        risk = f"Moderate/High risk — {malpractice_summary.get('count')} malpractice events logged."
    else:
        low = [k for k,v in numeric_subscores.items() if v < 4.0]
        risk = f"Weaknesses in {', '.join(low)}." if low else "Low risk — no major flags."

    strength = f"Top strength: {best_name} ({best_val:.1f}/10)."
    recommendation = ("Recommend: Strong hire / progress to onsite." if overall_cci >= 80 and malpractice_summary.get("deduction",0) < 3 else
                      "Recommend: Further technical interview or take-home task." if overall_cci >= 50 else
                      "Recommend: Not recommended at this time.")
    Badge = recommendation
    return {
        "trajectory": trajectory,
        "strength": strength,
        "risk": risk,
        "recommendation": recommendation,
        "Badge" : Badge,
    }

def evaluate_answer_with_gemini(question: str,
                                answer: str,
                                canonical_answer: str = "",
                                gemini_model: Optional[Any] = None,
                                verbose: bool = False) -> Dict[str, Any]:
    """
    Ask Gemini to grade a candidate's answer for a role-based question.

    Returns a dict:
      { "score": float (0-100) or None, "reason": str, "raw_llm_text": str }
    """
    model = gemini_model or genai.GenerativeModel(_GEMINI_MODEL_NAME)

    prompt = f"""
You are an experienced technical interviewer and grader.

Given:
- Question: "{question}"
- Candidate Answer: "{answer}"
- Canonical Ideal Answer: "{canonical_answer}"

Respond ONLY with a JSON object containing:
- score: integer 0-100 (how correct/complete the candidate answer is vs the canonical ideal)
- reason: one short sentence explaining the main justification for the score.

Example:
{{ "score": 85, "reason": "Candidate mentioned object pooling and profiling but missed memory detail." }}
"""

    raw_text = ""
    try:
        resp = model.generate_content(prompt)
        # extract text depending on sdk response shape
        raw_text = getattr(resp, "text", None) or (resp[0].content if isinstance(resp, (list, tuple)) and hasattr(resp[0], "content") else str(resp))
    except Exception as e:
        logger.exception("Gemini grading failed: %s", e)
        return {"score": None, "reason": f"LLM error: {e}", "raw_llm_text": raw_text}

    parsed_json = _safe_extract_json(raw_text or "")
    if not parsed_json:
        # if we couldn't parse JSON, return raw text in reason for debugging
        return {"score": None, "reason": f"No JSON from LLM: {raw_text[:300]}", "raw_llm_text": raw_text}

    score = parsed_json.get("score")
    reason = parsed_json.get("reason") or parsed_json.get("note") or ""
    try:
        if score is not None:
            score = float(score)
            score = max(0.0, min(100.0, score))
    except Exception:
        score = None

    return {"score": score, "reason": reason, "raw_llm_text": raw_text}
