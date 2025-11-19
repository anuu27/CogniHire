# src/airf/llm_eval.py
import json, textwrap
from typing import Dict, Any, List
from .llm_interface import GeminiAdapter
import math
import numpy as np
from .report_generator import generate_radar

# Strict, persona-driven prompt templates. These include micros and scoring rubrics.
QUESTION_PROMPT = textwrap.dedent("""
YOU ARE A RIGOROUS TECHNICAL EVALUATION PANEL composed of world-class academics (Terence Tao, Po-Shen Loh) and an IMO judge.
INSTRUCTIONS:
- You will score the candidate answer against the canonical answer on these five axes (0..10): conceptual, precision, clarity, reasoning, creativity.
- Be strict: 8 is attainable with solid correctness; 9+ is reserved for near-perfect, original, insightful answers.
- Provide a confidence between 0..1.
- Output STRICT JSON only (no extra commentary). Keys: conceptual, precision, clarity, reasoning, creativity, confidence, explanation.

CONTEXT:
Role brief: {role_brief}

CANONICAL:
{canonical}

CANDIDATE:
{candidate}

QUESTION:
{question}
""")

CV_PROMPT = textwrap.dedent("""
YOU ARE A RIGOROUS CV REVIEWER emulating a panel of academics and senior hiring judges.
INSTRUCTIONS:
- Score the CV on 5 metrics (0..10): target_alignment, impact_quantifier, trajectory_coherence, technical_resolution, career_velocity.
- Use the job role brief for target alignment.
- Provide a short JSON explanation field "notes".
- Strict scoring: penalize vague claims, reward quantified outcomes and concrete project details.
- Output STRICT JSON only.

ROLE_PROFILE:
{role_brief}

CV_JSON:
{cv_json}
""")

SYNTHESIS_PROMPT = textwrap.dedent("""
YOU ARE THE PANEL synthesizing the aggregated technical scores and CV metrics.
INSTRUCTIONS:
- Produce JSON with keys: TRAJECTORY, STRENGTH, RISK, RECOMMENDATION, SUMMARY_TEXT
- TRAJECTORY: one-line description of trend (stable/improving/declining)
- STRENGTH: comma-separated top strengths
- RISK: comma-separated top risks
- RECOMMENDATION: one-line recommended next step (e.g., specific follow-up task)
- SUMMARY_TEXT: 1-2 sentence human-quality summary
- Use the following input JSON blob literally: {blob}
- Output STRICT JSON only.
""")

def _parse_json_safe(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    # If it's already JSON
    try:
        return json.loads(text)
    except Exception:
        # Attempt to find first { ... } block
        start = text.find('{')
        end = text.rfind('}')
        if start!=-1 and end!=-1 and end>start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {}

def evaluate_questions_with_gemini(questions: List[Dict[str,Any]], role_brief: str, gemini: GeminiAdapter):
    per_q = []
    for q in questions:
        prompt = QUESTION_PROMPT.format(role_brief=role_brief,
                                       canonical=q.get("canonical_answer",""),
                                       candidate=q.get("candidate_answer",""),
                                       question=q.get("question",""))
        resp_text = gemini.generate(prompt, max_tokens=800, temperature=0.0)
        parsed = _parse_json_safe(resp_text)
        # normalize keys and fallbacks
        def _f(k): return float(parsed.get(k, 0.0))
        result = {
            "question_id": q.get("id"),
            "conceptual": round(_f("conceptual"),2),
            "precision": round(_f("precision"),2),
            "clarity": round(_f("clarity"),2),
            "reasoning": round(_f("reasoning"),2),
            "creativity": round(_f("creativity"),2),
            "confidence": float(parsed.get("confidence", 0.0)),
            "explanation": str(parsed.get("explanation","")),
            "raw": parsed
        }
        per_q.append(result)
    return per_q

def evaluate_cv_with_gemini(cv_json: Dict[str,Any], role_brief: str, gemini: GeminiAdapter):
    prompt = CV_PROMPT.format(role_brief=role_brief, cv_json=json.dumps(cv_json, ensure_ascii=False))
    resp_text = gemini.generate(prompt, max_tokens=800, temperature=0.0)
    parsed = _parse_json_safe(resp_text)
    # map to expected keys (0..10)
    out = {
        "target_alignment": float(parsed.get("target_alignment", parsed.get("relevance", 0.0))),
        "impact_quantifier": float(parsed.get("impact_quantifier", parsed.get("impact", 0.0))),
        "trajectory_coherence": float(parsed.get("trajectory_coherence", parsed.get("trajectory", 0.0))),
        "technical_resolution": float(parsed.get("technical_resolution", parsed.get("specificity", 0.0))),
        "career_velocity": float(parsed.get("career_velocity", parsed.get("velocity", 0.0))),
        "notes": parsed.get("notes", "")
    }
    return out

def synthesize_judgment(aggregated: Dict[str,Any], gemini: GeminiAdapter):
    blob = json.dumps(aggregated, ensure_ascii=False)
    prompt = SYNTHESIS_PROMPT.format(blob=blob)
    resp_text = gemini.generate(prompt, max_tokens=600, temperature=0.0)
    parsed = _parse_json_safe(resp_text)
    return {
        "TRAJECTORY": parsed.get("TRAJECTORY", parsed.get("trajectory","Stable")),
        "STRENGTH": parsed.get("STRENGTH", parsed.get("strength","")),
        "RISK": parsed.get("RISK", parsed.get("risk","")),
        "RECOMMENDATION": parsed.get("RECOMMENDATION", parsed.get("recommendation","")),
        "SUMMARY_TEXT": parsed.get("SUMMARY_TEXT", parsed.get("summary_text",""))
    }

def aggregate_and_score(questions: List[Dict[str,Any]], cv_json: Dict[str,Any], role_config: Dict[str,Any], gemini: GeminiAdapter):
    role_brief = role_config.get("role_brief", role_config.get("role_name","Role"))
    per_q = evaluate_questions_with_gemini(questions, role_brief, gemini)
    tech_aspects = ["conceptual","reasoning","precision","clarity","creativity"]
    aspect_avg = {}
    for a in tech_aspects:
        vals = [p.get(a,0.0) for p in per_q]
        aspect_avg[a] = float(round(sum(vals)/len(vals),2)) if vals else 0.0
    tech_avg = float(round(sum(aspect_avg.values())/len(aspect_avg) if aspect_avg else 0.0,2))
    cv_scores = evaluate_cv_with_gemini(cv_json, role_brief, gemini)
    cv_avg = float(round(sum([cv_scores[k] for k in ["target_alignment","impact_quantifier","trajectory_coherence","technical_resolution","career_velocity"]]) / 5.0, 2))
    alpha = float(role_config.get("alpha",0.75))
    CCI_raw = alpha * tech_avg + (1-alpha) * cv_avg
    CCI = round(max(0.0, min(10.0, CCI_raw)),2)
    badge = "WEAK"
    if CCI >= 8.0:
        badge = "HIRE-READY"
    elif CCI >= 6.5:
        badge = "GOOD POTENTIAL"
    synthesis_input = {
        "technical_summary": {"aspect_avg": aspect_avg, "technical_average": tech_avg},
        "cv_metrics": cv_scores,
        "CCI": CCI
    }
    synthesis = synthesize_judgment(synthesis_input, gemini)
    # generate radar
    labels = ["Conceptual","Reasoning","Precision","Clarity","Creativity"]
    values = [aspect_avg.get("conceptual",0.0), aspect_avg.get("reasoning",0.0), aspect_avg.get("precision",0.0), aspect_avg.get("clarity",0.0), aspect_avg.get("creativity",0.0)]
    generate_radar(labels, values, outpath="radar.png")
    return {
        "per_question": per_q,
        "technical_summary": {"aspect_avg": aspect_avg, "technical_average": tech_avg},
        "cv_metrics": cv_scores,
        "CCI": CCI,
        "badge": badge,
        "synthesis": synthesis
    }
