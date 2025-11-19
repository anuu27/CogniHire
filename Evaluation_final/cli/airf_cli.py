# cli/airf_cli.py
#!/usr/bin/env python3
import os, sys, json, subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.airf.llm_interface import GeminiAdapter
from src.airf.llm_eval import aggregate_and_score
from src.airf.report_generator import generate

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--cv", required=True)
    p.add_argument("--out", default="report.pdf")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args(argv)

    role = load_json(args.role)
    answers = load_json(args.answers)
    cv = load_json(args.cv)

    # Always a list
    if isinstance(answers, dict) and "questions" in answers:
        answer_list = answers["questions"]
    else:
        answer_list = answers

    gemini = GeminiAdapter()

    # LLM-driven scoring pipeline
    res = aggregate_and_score(answer_list, cv, role, gemini)

    # Build aggregate dict for report generation
    agg = {
        "role_name": role.get("role_name", "Role"),
        "candidate": cv.get("candidate_name", "Candidate"),
        "per_question": res["per_question"],
        "technical_summary": res["technical_summary"],
        "cv_metrics": res["cv_metrics"],
        "CCI": res["CCI"],
        "badge": res["badge"],
        "summary_text": res["synthesis"].get("SUMMARY_TEXT", ""),
        "recommendation": res["synthesis"].get("RECOMMENDATION", ""),
        "trajectory": {"label": res["synthesis"].get("TRAJECTORY", "")},
        "risk": res["synthesis"].get("RISK", ""),
        "confidence": res.get("confidence", 0.0)
    }

    # Normalized CV scores for report
    cv_scores = {
        "relevance": res["cv_metrics"].get("target_alignment", 0.0) / 10.0,
        "plausibility": res["cv_metrics"].get("trajectory_coherence", 0.0) / 10.0,
        "specificity": res["cv_metrics"].get("technical_resolution", 0.0) / 10.0,
        "consistency": res["cv_metrics"].get("trajectory_coherence", 0.0) / 10.0,
    }

    texfile = generate(agg, cv_scores, res["per_question"])
    print("Wrote tex:", texfile)

    if not args.no_compile:
        try:
            subprocess.run(
                ["xelatex", "-interaction=nonstopmode", texfile],
                check=True
            )
            pdfname = texfile.replace(".tex", ".pdf")
            if os.path.exists(pdfname):
                os.replace(pdfname, args.out)
                print("Wrote PDF:", args.out)
        except Exception as e:
            print("PDF compilation failed:", e)

if __name__ == "__main__":
    main()

