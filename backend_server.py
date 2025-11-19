# backend_server.py
import os
import json
import tempfile
import subprocess
import shutil
import io
import matplotlib
matplotlib.use("Agg")# headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory,send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from interview_manager import InterviewManager  
from evaluation import evaluate_interview
from math import pi
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import tempfile 
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
from flask import make_response





Path("data/malpractice_logs").mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow requests from your frontend (http://localhost:5173)

# Keep a simple in-memory mapping of session_id -> InterviewManager instance
SESSIONS = {}
MALPRACTICE_LOGS = {}
def map_role_to_file(role: str) -> str:
    """
    Convert user-friendly role names to canonical JSON filenames.
    Example: "AI Engineer" -> "ai_engineer.json"
    """
    if not role:
        return None

    # normalize
    key = role.strip().lower().replace(" ", "_")

    return f"{key}.json"


import io
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from flask import send_file

# Helper: create radar chart
def _make_radar_chart(report, out_path):
    """Create a radar chart image from report['components']."""
    comps = report.get("components") or {}
    # prefer readable labels
    labels = ["role_relevance", "canonical_component", "resume_component", "completeness", "conciseness", "fluency"]
    values = [float(comps.get(k) or 0.0) for k in labels]

    # If values appear to be on 0-100 scale, convert to 0-10
    max_v = max(values) if values else 1.0
    if max_v > 10.0:
        values = [v * 10.0 / max_v for v in values]

    n = len(labels)
    angles = [2 * pi * i / n for i in range(n)]
    angles += angles[:1]
    vals = values + values[:1]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180/pi for a in angles[:-1]], labels, fontsize=10)
    ax.plot(angles, vals, linewidth=2, linestyle='solid', label="Score")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_ylim(0, max(10, max(vals)))
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Helper: create per-question horizontal bars image
def _make_per_question_bars(report, out_path):
    """Create a horizontal bar image representing per-question canonical_score or canonical_similarity."""
    arr = report.get("per_question") or []
    if not arr:
        fig = plt.figure(figsize=(6,1.5))
        plt.text(0.5, 0.5, "No per-question data", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    labels = []
    scores = []
    for q in arr:
        # Use short label: first 30 chars of question or the type
        qtext = (q.get("question") or "").strip()
        label = (q.get("type") or "").strip() or (qtext[:28] + ("…" if len(qtext) > 28 else ""))
        labels.append(label)
        sc = q.get("canonical_score")
        if sc is None:
            sc = (q.get("canonical_similarity") or 0.0) * 100.0
        try:
            scores.append(float(sc))
        except Exception:
            scores.append(0.0)

    # keep max width sensible
    height = max(1.6, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(6, height))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, scores, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_xlim(0, max(100, max(scores) + 5))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Minimal LaTeX escaping helper
# latex_template.py (or add to your evaluation/report generator module)
def _latex_escape(s: str):
    if s is None:
        return ""
    return str(s).replace("\\", "\\\\").replace("&", "\\&").replace("%", "\\%")\
                 .replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")\
                 .replace("{", "\\{").replace("}", "\\}").replace("~", "\\textasciitilde{}")\
                 .replace("^", "\\^{}")

LATEX_TEMPLATE = r'''
\documentclass[10pt]{article}
\usepackage[a4paper,margin=0.7in]{geometry}
\usepackage{fontspec}
\usepackage[HTML]{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{microtype}
\usepackage{enumitem}
\setmainfont{JetBrains Mono}

\definecolor{taobg}{HTML}{0f0f0f}
\definecolor{taogreen}{HTML}{50F0AA}
\definecolor{taopink}{HTML}{FF7796}
\definecolor{taogray}{HTML}{3A3A3A}
\definecolor{taowhite}{HTML}{D0F0E0}
\pagecolor{taobg}
\color{taogreen}
\raggedright
\setlength{\parskip}{6pt}
\newcommand{\baritem}[3]{%
  \noindent{\color{taopink}\texttt{-- #1:}}%
  \begin{tikzpicture}[baseline=-0.5ex]
    \fill[taogray!35] (0,0) rectangle (6,0.28);
    \pgfmathsetmacro{\val}{min(max(#2,0),10)}
    \fill[taogreen] (0,0) rectangle (6*\val/10,0.28);
    \draw[taogray!60] (6*\val/10,0) rectangle (6,0.28);
  \end{tikzpicture}%
  \hspace{0.6em}{\color{taopink}\texttt{#3}}%
}

\begin{document}
\thispagestyle{empty}

{\bfseries AIRF Evaluation Report — {{ROLE_NAME}}} \\[6pt]
\textcolor{taopink}{Candidate: [{{CANDIDATE}}]} \\[6pt]

\textbf{Cognitive Competency Index (CCI):} \textcolor{taogreen}{\textbf{{CCI}} — {{BADGE}}} \\[6pt]
Tech Avg: {{TECH_AVG_10}} \quad CV Score: {{CV_RELEVANCE}} \quad Confidence (σ): {{SIGMA}} \\[12pt]

\section*{Numeric Subscores (0--10)}
\baritem{Conceptual correctness}{ {{S_CONCEPTUAL}} }{ {{S_CONCEPTUAL}} }\\[4pt]
\baritem{Reasoning depth}{ {{S_REASONING}} }{ {{S_REASONING}} }\\[4pt]
\baritem{Precision grounding}{ {{S_PRECISION}} }{ {{S_PRECISION}} }\\[4pt]
\baritem{Communication clarity}{ {{S_CLARITY}} }{ {{S_CLARITY}} }\\[4pt]
\baritem{Creativity / insight}{ {{S_CREATIVITY}} }{ {{S_CREATIVITY}} }\\[12pt]

\section*{Human Summary}
\noindent\textbf{Trajectory:} {{TRAJECTORY}} \\[4pt]
\noindent\textbf{Strength:} {{STRENGTH}} \\[4pt]
\noindent\textbf{Risk:} {{RISK}} \\[4pt]
\noindent\textbf{Recommendation:} {{RECOMMENDATION}} \\[12pt]

\section*{Per-question Summary}
\begin{enumerate}[leftmargin=*]
{{PER_QUESTION_ITEMS}}
\end{enumerate}

\section*{Malpractice Summary}
\noindent Count: {{MALP_COUNT}} \\
\noindent Types: {{MALP_TYPES}} \\
\noindent Deduction: {{MALP_DEDUCTION}} \\[8pt]

\begin{center}
\includegraphics[width=0.6\textwidth]{{RADAR_PATH}}
\end{center}

\vfill
\noindent\textbf{Evaluator Note:} \\
{{EVALUATOR_NOTE}}

\end{document}
'''

# Try to build via pdflatex/xelatex in a tempdir; return Path to PDF on success else None
def _try_build_pdf_with_latex(tex_content: str, workdir: Path) -> Path:
    tex_file = workdir / "report.tex"
    tex_file.write_text(tex_content, encoding="utf-8")
    # try xelatex first (for fontspec), else pdflatex
    candidates = [["xelatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_file.name)],
                  ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_file.name)]]
    for cmd in candidates:
        try:
            subprocess.run(cmd, cwd=str(workdir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            # run twice for crossrefs
            subprocess.run(cmd, cwd=str(workdir), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            pdf_path = workdir / "report.pdf"
            if pdf_path.exists():
                return pdf_path
        except Exception:
            # try next candidate
            continue
    return None

# ReportLab fallback builder
def _build_pdf_with_reportlab(report, radar_path: Path, bars_path: Path, out_pdf_path: Path):
    c = canvas.Canvas(str(out_pdf_path), pagesize=A4)
    width, height = A4
    margin = 18 * mm
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "AIRF Competency Report")
    y -= 14

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Candidate: {report.get('Name','')}")
    y -= 12
    c.drawString(margin, y, f"Role: {report.get('role','')}")
    y -= 12
    c.drawString(margin, y, f"Date: {report.get('end_time','')}")
    y -= 16

    # Scores block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Final CIS: {report.get('cci',{}).get('raw','N/A')} / 100")
    y -= 14
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Overall Score: {report.get('overall_score','N/A')}")
    y -= 16

    # Per-question bars image
    

    # Radar
    try:
        if radar_path.exists():
            img = ImageReader(str(radar_path))
            # reserve area
            img_h = 120*mm
            if y - img_h < margin:
                c.showPage()
                y = height - margin
            c.drawImage(img, margin, y - img_h, width=width - 2*margin, height=img_h, preserveAspectRatio=True)
            y -= img_h + 8
    except Exception:
        pass

    # Components / aspect breakdown
    comps = report.get("components") or {}
    subs = report.get("subscores") or {}
    if y < (margin + 120):
        c.showPage()
        y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Aspect Breakdown")
    y -= 14
    c.setFont("Helvetica", 11)
    lines = [
        ("Conceptual correctness", subs.get("Conceptual correctness", "N/A")),
        ("Reasoning depth",subs.get("Reasoning depth", "N/A")),
        ("Precision grounding", subs.get("Precision grounding", "N/A")),
        ("Communication Clarity", subs.get("Communication clarity", "N/A")),
        ("Creativity/Insight", subs.get("Creativity / insight", "N/A")),
    ]
    for name, val in lines:
        if y < margin + 30:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, f"{name}: {val}")
        y -= 12

    # Malpractice summary
    mal = report.get("malpractice_detected") or {}
    if y < margin + 60:
        c.showPage()
        y = height - margin
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Malpractice & Risk")
    y -= 14
    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"Flags: {mal.get('count',0)}  Types: {', '.join(mal.get('types',[]) if isinstance(mal.get('types',[]), list) else [str(mal.get('types'))])}")
    y -= 14
    c.drawString(margin, y, f"Deduction: {mal.get('deduction', 0)}")
    y -= 18

    # Human summary fields if present
    if report.get("trajectory") or report.get("strength") or report.get("risk") or report.get("recommendation") or report.get("Badge"):
        # helper to wrap text to lines that fit width using canvas.stringWidth
        def wrap_text_to_lines(text, font_name, font_size, max_width, c):
            words = str(text).split()
            lines = []
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if c.stringWidth(test, font_name, font_size) <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            return lines

        # page guard
        if y < margin + 80:
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Human Summary")
        y -= 14

        # keys fixed (each a separate key)
        keys = ("trajectory", "strength", "risk", "recommendation", "Badge")
        font_name = "Helvetica"
        font_size = 10
        leading = 12  # vertical space per line

        for key in keys:
            val = report.get(key)
            if not val:
                continue

            # draw key header
            header = key.capitalize() + ":"
            # if header would overflow page, new page
            if y - leading < margin:
                c.showPage()
                y = height - margin

            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, header)
            y -= leading

            # wrap the value into lines that fit inside page width
            max_text_width = width - 2 * margin
            c.setFont(font_name, font_size)
            lines = wrap_text_to_lines(val, font_name, font_size, max_text_width, c)

            # start writing lines, paginating as needed
            for ln in lines:
                if y - leading < margin:
                    # flush current page and start new
                    c.showPage()
                    y = height - margin
                    # re-draw section header on new page? (optional)
                c.drawString(margin, y, ln)
                y -= leading

            # small spacing after each field
            y -= 6

    c.showPage()
    c.save()

# Flask endpoint: generate and return PDF bytes
@app.get("/api/employer/report_pdf/<session_id>")
def get_report_pdf(session_id):
    reports_dir = Path("data/interviews")
    file = reports_dir / f"{session_id}_report.json"
    if not file.exists():
        app.logger.warning("report_pdf: report json not found for %s", session_id)
        return jsonify({"error": "Report not found"}), 404

    try:
        report = json.loads(file.read_text(encoding="utf-8"))
    except Exception as e:
        app.logger.exception("report_pdf: failed to parse report json for %s: %s", session_id, e)
        return jsonify({"error": "Failed to parse report JSON", "detail": str(e)}), 500

    try:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)
            radar_path = workdir / "radar.png"
            bars_path = workdir / "bars.png"
            pdf_out = workdir / "report.pdf"

            # generate visual assets
            try:
                _make_radar_chart(report, str(radar_path))
            except Exception:
                app.logger.exception("report_pdf: radar generation failed for %s", session_id)
            try:
                _make_per_question_bars(report, str(bars_path))
            except Exception:
                app.logger.exception("report_pdf: per-question bars generation failed for %s", session_id)

            # Attempt LaTeX build (if you want to use LATEX_TEMPLATE defined earlier in your file)
            pdf_path = None
            try:
                # if you defined LATEX_TEMPLATE earlier, populate it here (light-weight fill)
                if 'LATEX_TEMPLATE' in globals():
                    tex = LATEX_TEMPLATE
                    tex = tex.replace("{{CANDIDATE}}", _latex_escape(report.get("Name", report.get("candidate_name", ""))))
                    tex = tex.replace("{{ROLE_FULL}}", _latex_escape(report.get("role", "")))
                    tex = tex.replace("{{DATE}}", _latex_escape(report.get("end_time", "")))
                    tex = tex.replace("{{CCI}}", str(report.get("cci", {}).get("raw", "")))
                    tex = tex.replace("{{SIGMA}}", "N/A")
                    tex = tex.replace("{{BADGE}}", _latex_escape(report.get("badge") or ""))
                    tex = tex.replace("{{TECH_AVG}}", _latex_escape(str(round((report.get("overall_score") or 0)/10.0, 2))))
                    tex = tex.replace("{{PER_QUESTION_BARS}}", "See per-question chart included.")
                    # copy radar if exists
                    if radar_path.exists():
                        (workdir / "radar_for_tex.png").write_bytes(radar_path.read_bytes())
                        tex = tex.replace("{{RADAR_PATH}}", "radar_for_tex.png")
                    else:
                        tex = tex.replace("{{RADAR_PATH}}", "")

                    try:
                        pdf_path = _try_build_pdf_with_latex(tex, workdir)
                    except Exception:
                        app.logger.exception("report_pdf: latex compile attempt raised exception for %s", session_id)
                        pdf_path = None
            except Exception:
                app.logger.exception("report_pdf: preparing latex content failed for %s", session_id)
                pdf_path = None

            # fallback to reportlab if latex unsuccessful
            if not pdf_path:
                try:
                    _build_pdf_with_reportlab(report, radar_path, bars_path, pdf_out)
                    if pdf_out.exists():
                        pdf_path = pdf_out
                except Exception:
                    app.logger.exception("report_pdf: reportlab fallback failed for %s", session_id)
                    pdf_path = None

            if not pdf_path or not pdf_path.exists():
                app.logger.error("report_pdf: PDF not created for %s", session_id)
                return jsonify({"error": "PDF generation failed"}), 500

            # read bytes into memory (avoid locking temp files on Windows)
            pdf_bytes = pdf_path.read_bytes()

            # optionally save a copy
            try:
                saved_pdf = reports_dir / f"{session_id}_report.pdf"
                saved_pdf.write_bytes(pdf_bytes)
                app.logger.info("report_pdf: saved copy to %s", saved_pdf)
            except Exception:
                app.logger.exception("report_pdf: could not persist PDF copy for %s", session_id)

            return send_file(
                io.BytesIO(pdf_bytes),
                as_attachment=True,
                download_name=f"{session_id}_report.pdf",
                mimetype="application/pdf"
            )
    except Exception as e:
        app.logger.exception("report_pdf: unexpected error for %s: %s", session_id, e)
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route("/api/upload_resume", methods=["POST"])
def upload_resume():
    # Accepts multipart/form-data with field 'resume'
    f = request.files.get("resume")
    if not f:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(f.filename)
    save_path = UPLOAD_DIR / filename
    f.save(save_path)

    # (Optional) parse resume with your QuestionGenerator if you have that accessible:
    # Here we return a minimal parsed response so frontend shows something.
    parsed = {"filename": filename}
    return jsonify({"parsed": parsed})



@app.route("/api/start_interview", methods=["POST"])
def start_interview():
    # read form values
    resume_path = request.form.get("resume_path")
    role = request.form.get("role", "").strip()

    # basic validation
    if not resume_path:
        return jsonify({"error": "resume_path missing"}), 400
    if not role:
        return jsonify({"error": "role missing"}), 400

    # build full resume filesystem path (uploads folder)
    resume_file = UPLOAD_DIR / secure_filename(resume_path)
    if not resume_file.exists():
        # helpful debug message
        return jsonify({"error": f"Resume not found on server: {resume_file}"}), 404

    try:
        # Map role to filename if you use role-file mapping (optional)
        # role_file = map_role_to_file(role)  # if using mapping helper

        # Create InterviewManager with the role (it should use role to load role-questions)
        im = InterviewManager(str(resume_file), role)
        SESSIONS[im.session_id] = im

        # Start interview (returns opening plus first real Q)
        initial_items = im.start_interview()  # this should return a list
        first_question = initial_items[-1] if initial_items else {"question": "No question", "type": "end"}

        # Log for debugging
       
        return jsonify({"session_id": im.session_id, "first_question": first_question})

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("start_interview failed: %s\n%s", e, tb)
        return make_response(jsonify({"error": str(e), "traceback": tb}), 500)



@app.route("/api/log_malpractice", methods=["POST"])
def log_malpractice():
    
    raw = request.get_data(as_text=True)
    

    data = None
    try:
        data = request.get_json(silent=True)
    except Exception as e:
        app.logger.exception("get_json error: %s", e)

    if not data:
        # fallback: try to json.loads raw
        try:
            data = json.loads(raw) if raw else None
        except Exception as e:
            app.logger.exception("raw json parse failed: %s", e)
            return jsonify({"error": "Invalid JSON"}), 400

    # Normalize session_id
    session_id = (data.get("session_id") or "").strip()
    event = data.get("event") or {}

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if not event or not isinstance(event, dict):
        return jsonify({"error": "Missing or invalid event object"}), 400

    # Ensure data/malpractice_logs dir exists (use same directory used by end_interview)
    base_dir = Path("data") / "malpractice_logs"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save in-memory
    MALPRACTICE_LOGS.setdefault(session_id, []).append(event)

    # 2) Append per-session JSONL (append-only)
    jsonl_path = base_dir / f"{session_id}_malpractice.jsonl"
    record = {"session_id": session_id, "event": event, "server_ts": datetime.utcnow().isoformat() + "Z"}
    try:
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
       
    except Exception as e:
        app.logger.exception("Failed to append JSONL: %s", e)

    # 3) Also update a human-readable JSON snapshot for end_interview and UI (overwrites)
    snapshot_path = base_dir / f"{session_id}_malpractice.json"
    try:
        # Dump the in-memory list (so snapshot always reflects current in-memory state)
        with open(snapshot_path, "w", encoding="utf-8") as fh:
            json.dump(MALPRACTICE_LOGS.get(session_id, []), fh, ensure_ascii=False, indent=2)
        
    except Exception as e:
        app.logger.exception("Failed to write JSON snapshot: %s", e)

    total_now = len(MALPRACTICE_LOGS.get(session_id, []))
    

    return jsonify({"success": True, "session_id": session_id, "count": total_now}), 200

@app.route("/api/download_transcript/<session_id>", methods=["GET"])
def download_transcript(session_id):
    # Return saved transcript file if exists
    path = Path("data/interviews") / f"{session_id}_transcript.json"
    if path.exists():
        return send_from_directory(path.parent.resolve(), path.name, as_attachment=True)
    return jsonify({"error": "Transcript not found"}), 404


@app.route("/api/end_interview", methods=["POST"])
def end_interview():
    session_id = request.form.get("session_id")
    if session_id not in SESSIONS:
        return jsonify({"error":"Unknown session_id"}), 404

    im = SESSIONS[session_id]
    try:
        # 1. Save Transcript (Will save as a dictionary, confirmed by your code)
        im.save_transcript()
        transcript_path = Path("data/interviews") / f"{im.session_id}_transcript.json"
        
        # 2. Handle Malpractice Logging
        malpractice_path = Path("data/malpractice_logs") / f"{im.session_id}_malpractice.json"
        malpractice_data = MALPRACTICE_LOGS.pop(im.session_id, [])
        malpractice_path.write_text(json.dumps(malpractice_data, indent=2), encoding="utf-8")
        
        # 3. Load and Extract Transcript List
        transcript_raw = transcript_path.read_text(encoding="utf-8")
        transcript_data = json.loads(transcript_raw)

        # FIX: Your save_transcript function makes transcript_data a DICT. 
        # We use .get() here to safely extract the list.
        if isinstance(transcript_data, dict):
            transcript_list = transcript_data.get("interview_transcript", [])
        elif isinstance(transcript_data, list):
            # Fallback for unexpected format (should not happen based on your save_transcript)
            transcript_list = transcript_data
        else:
            transcript_list = []
            app.logger.error("Transcript data for %s is neither a list nor a dict.", session_id)

        # 4. Run Evaluation
        report = evaluate_interview(session_id=im.session_id,
                                    Name = transcript_data.get("candidate_info", {}).get("name", "Unknown"),
            transcript_log = transcript_list, 
            role=im.role, 
            malpractice_events=malpractice_data 
        )
        
        # 5. Finalize Report
        report['raw_malpractice_log'] = malpractice_data
        report['malpractice_count'] = len(malpractice_data)
        
        rep_path = transcript_path.with_name(f"{im.session_id}_report.json")
        rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        
        SESSIONS.pop(session_id)
        
        return jsonify({
            "saved": True, 
            "transcript_path": str(transcript_path), 
            "report_path": str(rep_path),
            "malpractice_count": len(malpractice_data) 
        })
    except Exception as e:
        # Crucially, log the full traceback so you can find the exact line causing the 500
        app.logger.error("end_interview failed (Critical Error): %s", traceback.format_exc())
        return jsonify({"error": f"Critical server error during end interview. Check server logs: {str(e)}", "traceback": traceback.format_exc()}), 500



@app.route("/api/roles", methods=["GET"])
def get_roles():
    role_dir = Path("data/role_questions")
    roles = []

    for file in role_dir.glob("*.json"):
        # convert filename like "data_engineer.json" → "data engineer"
        name = file.stem.replace("_", " ")
        roles.append(name)

    # sort alphabetically
    roles.sort()

    return jsonify({"roles": roles})

import traceback

@app.route("/api/process_answer", methods=["POST"])
def process_answer():
    # Log full request for debugging
   

    session_id_raw = request.form.get("session_id")
    session_id = session_id_raw.strip() if isinstance(session_id_raw, str) else None
  
    if not session_id:
        return jsonify({"error": "Missing session_id in form data"}), 400

    if session_id not in SESSIONS:
        # Helpful debug: list available sessions (small)
        known = list(SESSIONS.keys())[:10]
        app.logger.warning("session_id not found in SESSIONS. Received: %s ; Known (first 10): %s", session_id, known)
        return jsonify({"error": "Invalid session_id", "received": session_id, "known_sample": known}), 404

    im = SESSIONS[session_id]

    f = request.files.get("answer_blob")
    if not f:
        app.logger.warning("No answer_blob provided for session %s", session_id)
        return jsonify({"error": "No answer_blob provided"}), 400

    filename = secure_filename(f.filename or "answer.webm")
    save_path = UPLOAD_DIR / f"{session_id}_{filename}"
    try:
        f.save(save_path)
    except Exception as e:
        app.logger.exception("Failed to save uploaded file")
        return jsonify({"error": f"Failed to save file: {e}"}), 500

    try:
        next_items = im.process_answer(str(save_path))
        next_q = next_items[0] if (isinstance(next_items, list) and next_items) else next_items
        return jsonify({"next_question": next_q})
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("process_answer failed: %s\n%s", e, tb)
        return jsonify({"error": str(e), "traceback": tb}), 500
@app.route("/api/employer/sessions", methods=["GET"])
def list_sessions():
    # scan data/interviews for *_transcript.json and *_report.json
    base = Path("data/interviews")
    sessions = []
    for p in base.glob("*_transcript.json"):
        sid = p.name.replace("_transcript.json","")
        report = base / f"{sid}_report.json"
        score = None
        try:
            if report.exists():
                score = json.loads(report.read_text()).get("overall_score")
        except: score = None
        sessions.append({"session_id": sid, "transcript_path": str(p), "report_path": str(report) if report.exists() else None, "score": score})
    # sort by date descending (you can parse start_time from transcript later)
    sessions.sort(key=lambda x: x["session_id"], reverse=True)
    return jsonify({"sessions": sessions})

@app.route("/api/employer/session/<session_id>", methods=["GET"])
def get_session(session_id):
    t = Path("data/interviews") / f"{session_id}_transcript.json"
    r = Path("data/interviews") / f"{session_id}_report.json"
    if not t.exists():
        return jsonify({"error":"session not found"}), 404
    transcript = json.loads(t.read_text())
    report = json.loads(r.read_text()) if r.exists() else None
    return jsonify({"transcript": transcript, "report": report})
@app.get("/api/employer/reports")
def list_reports():
    reports_dir = Path("data/interviews")
    output = []

    for file in reports_dir.glob("*_report.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            output.append({
                "session_id": data.get("session_id"),
                "role": data.get("role"),
                "overall_score": data.get("overall_score"),
                "cci": data.get("cci"),
                "candidate_name": data.get("Name", "Unknown"),
                "timestamp": data.get("end_time", "")
            })
        except Exception:
            continue
    
    return {"reports": output}
@app.get("/api/employer/report/<session_id>")
def get_report(session_id):
    file = Path("data/interviews") / f"{session_id}_report.json"
    if not file.exists():
        return {"error": "Report not found"}, 404

    data = json.loads(file.read_text(encoding="utf-8"))
    return data



if __name__ == "__main__":
    # Run on port 8000 without the autoreloader to avoid restarts
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)

