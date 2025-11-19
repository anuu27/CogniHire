// src/components/ReportDetail.jsx
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

export default function ReportDetail({ report: propReport = null, onDownload = null }) {
  const { sessionId } = useParams();
  const [report, setReport] = useState(propReport);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    async function fetchReportById(id) {
      if (!id) { setError("No session id provided."); return; }
      setLoading(true);
      try {
        const res = await fetch(`/api/employer/report/${id}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load report");
        if (mounted) setReport(data);
      } catch (err) {
        if (mounted) setError(err.message || String(err));
      } finally {
        if (mounted) setLoading(false);
      }
    }

    if (!propReport) {
      if (sessionId) fetchReportById(sessionId);
    } else {
      setReport(propReport);
    }

    return () => { mounted = false; };
  }, [propReport, sessionId]);

  useEffect(() => { if (propReport) setReport(propReport); }, [propReport]);

  if (error) return <div className="p-4 text-red-600">Error: {error}</div>;
  if (loading && !report) return <div className="p-4">Loading report...</div>;
  if (!report) return <div className="p-4">No report available.</div>;

  const session_id = report.session_id || report.sessionId || sessionId || "unknown";
  const candidate_name = report.Name || report.candidate_name || report.candidateName || "Unknown";
  const role = report.role || "Unknown";
  const resume_file = report.resume_file || report.resume_path || "";
  const overall_score = report.overall_score ?? report.score ?? null;
  const cci = report.cci || null;
  const components = report.components || null;
  const per_question = Array.isArray(report.per_question) ? report.per_question : (Array.isArray(report.questions) ? report.questions : []);
  const raw_malpractice_log = report.raw_malpractice_log || report.malpractice_events || [];
  const malpractice_detected = report.malpractice_detected || { count: raw_malpractice_log.length, types: [], deduction: 0 };
  const subscores = report.subscores || {};

  async function downloadPdf() {
    try {
      const res = await fetch(`/api/employer/report_pdf/${session_id}`);
      if (!res.ok) {
        const err = await res.json().catch(()=>({error:"PDF generation failed"}));
        alert("PDF generation failed: " + (err.error || JSON.stringify(err)));
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${session_id}_report.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Failed to download PDF: " + e.message);
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold">Interview Report</h1>
          <div className="text-sm text-gray-500 mt-1">Session: <strong>{session_id}</strong></div>
          <div className="text-sm text-gray-500 mt-1">Candidate: <strong>{candidate_name}</strong></div>
        </div>

        <div className="flex gap-2">
          <button onClick={downloadPdf} className="bg-indigo-600 text-white px-3 py-2 rounded">Download PDF</button>
          {onDownload && <button onClick={onDownload} className="bg-gray-200 px-3 py-2 rounded">Download JSON</button>}
        </div>
      </div>

      {/* SESSION INFO */}
      <div className="p-4 bg-white rounded-xl shadow">
        <h2 className="text-2xl font-semibold mb-2">Session Details</h2>

        <p><strong>Session ID:</strong> {session_id}</p>
        <p><strong>Role Applied:</strong> {role}</p>
        <p><strong>Resume File:</strong> {resume_file}</p>

        {report.resume_info_evaluated && Object.keys(report.resume_info_evaluated).length > 0 && (
          <div className="mt-3">
            <h3 className="text-lg font-semibold">Candidate Info (Extracted from Resume)</h3>
            <pre className="bg-gray-100 p-3 rounded-md whitespace-pre-wrap">
              {JSON.stringify(report.resume_info_evaluated, null, 2)}
            </pre>
          </div>
        )}
      </div>

      {/* SUBSCORES */}
      <div className="p-4 bg-white rounded-xl shadow">
        <h2 className="text-2xl font-semibold mb-2">Subscores (0-10)</h2>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(subscores).map(([k,v]) => (
            <div key={k} className="p-3 border rounded">
              <div className="flex justify-between items-baseline">
                <div className="font-medium">{k}</div>
                <div className="text-sm text-gray-600">{typeof v === "number" ? v.toFixed(1) : v}</div>
              </div>
              <div className="w-full bg-gray-100 h-2 rounded mt-2">
                <div style={{width: `${Math.min(100, (parseFloat(v)||0)/10*100)}%`}} className="h-2 rounded bg-green-400" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* HUMAN SUMMARIES */}
      <div className="p-4 bg-white rounded-xl shadow">
        <h2 className="text-2xl font-semibold mb-2">Human Summary</h2>
        <p><strong>Trajectory:</strong> {report.trajectory || "—"}</p>
        <p><strong>Strength:</strong> {report.strength || "—"}</p>
        <p><strong>Risk:</strong> {report.risk || "—"}</p>
        <p><strong>Recommendation:</strong> {report.recommendation || "—"}</p>
      </div>

      {/* SECURITY FLAGS */}
      <div className="p-4 bg-red-50 border border-red-300 rounded-xl">
        <h2 className="text-2xl font-bold text-red-700 mb-2">Security / Behaviour Flags</h2>

        {malpractice_detected && malpractice_detected.count > 0 ? (
          <div>
            <p className="mb-2">Total flags: <strong>{malpractice_detected.count}</strong> • Deduction: <strong>{malpractice_detected.deduction}</strong></p>
            <ul className="list-disc ml-6">
              {(malpractice_detected.types || []).map((t, i) => <li key={i}>{t}</li>)}
            </ul>

            <div className="mt-3">
              <h3 className="font-semibold">Raw Malpractice Events</h3>
              <pre className="bg-white p-3 rounded-md whitespace-pre-wrap text-sm">
                {JSON.stringify(raw_malpractice_log, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <p className="text-gray-600">No flags recorded.</p>
        )}
      </div>

      {/* SCORES */}
      <div className="p-4 bg-white rounded-xl shadow">
        <h2 className="text-2xl font-semibold mb-2">Overall Evaluation</h2>

        <p><strong>Overall Score:</strong> {overall_score ?? "N/A"}</p>
        {report.score_before_malpractice != null && <p><strong>Score before malpractice:</strong> {report.score_before_malpractice}</p>}

        {cci && (
          <>
            <p><strong>CCI Raw:</strong> {cci.raw ?? "N/A"}</p>
            <p><strong>Percentile (vs role):</strong> {cci.percentile_vs_role ?? "N/A"}%</p>
          </>
        )}
      </div>

     {/* PER QUESTION SECTION */}
<div className="p-4 bg-white rounded-xl shadow space-y-4">
  <h2 className="text-2xl font-semibold mb-4">Per-Question Analysis</h2>

  {per_question && per_question.length > 0 ? (
    per_question.map((q, index) => {
      const type = (q.type || "standard").toLowerCase();

      // Helper small renderers
      const renderCanonical = () => (
        <>
          {q.canonical_answer && (
            <p className="text-sm text-gray-700 mt-2">
              <strong>Expected Answer:</strong> {q.canonical_answer}
            </p>
          )}
          <p className="mt-2"><strong>Canonical similarity:</strong> {q.canonical_similarity ?? "N/A"}</p>
          <p><strong>Canonical score:</strong> {q.canonical_score ?? "N/A"}</p>
          {/* If you added Gemini, show its fields if present */}
          {q.gemini_score != null && <p><strong>Gemini score:</strong> {q.gemini_score}</p>}
          {q.combined_role_score != null && <p><strong>Combined role score:</strong> {q.combined_role_score}</p>}
          <p><strong>Q/A similarity:</strong> {q.q_a_similarity ?? "N/A"}</p>
          <p><strong>Answer length:</strong> {q.answer_length ?? 0}</p>
        </>
      );

      const renderLLMEval = (evalObj) => {
        if (!evalObj) return <p className="text-gray-500">No LLM evaluation available.</p>;
        // try common keys used by evaluate_resume_answer_with_llm
        const score = evalObj.resume_score ?? evalObj.score ?? null;
        const evidence = evalObj.evidence || evalObj.evidence_found || [];
        const weaknesses = evalObj.weaknesses || [];
        const notes = evalObj.notes || evalObj.reason || "";
        const raw = evalObj.raw_llm_text || evalObj.raw || "";

        return (
          <div className="mt-2 p-3 bg-white border rounded">
            <p><strong>LLM evaluation</strong></p>
            {score != null && <p><strong>Score:</strong> {Math.round(Number(score) * 100) / 100}</p>}
            {notes && <p className="text-sm text-gray-700"><strong>Notes:</strong> {notes}</p>}
            {evidence && evidence.length > 0 && (
              <div className="mt-2">
                <div className="font-semibold text-sm">Evidence</div>
                <ul className="list-disc ml-6 text-sm">
                  {evidence.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
              </div>
            )}
            {weaknesses && weaknesses.length > 0 && (
              <div className="mt-2">
                <div className="font-semibold text-sm">Weaknesses</div>
                <ul className="list-disc ml-6 text-sm">
                  {weaknesses.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}
            {raw && (
              <details className="mt-2 text-xs text-gray-500">
                <summary>Raw LLM output</summary>
                <pre className="whitespace-pre-wrap mt-2">{raw}</pre>
              </details>
            )}
          </div>
        );
      };

      return (
        <div key={index} className="border p-3 rounded-xl bg-gray-50">
          <p><strong>Type:</strong> {q.type ?? "standard"}</p>
          <p className="mt-1"><strong>Question:</strong> {q.question ?? "—"}</p>
          <p className="mt-1"><strong>Answer:</strong> {q.answer || "—"}</p>

          {type === "role" && (
            <>
              {renderCanonical()}
            </>
          )}

          {(type === "resume" || type === "follow-up" || type === "followup") && (
            <>
              <div className="mt-3">
                {/* Prefer the evaluator combined object, or fall back to q.resume_eval */}
                {renderLLMEval(q.resume_eval || q.llm_eval || q.evaluation)}
              </div>

              {/* still show some similarity if available */}
              <div className="mt-3">
                <p><strong>Q/A similarity:</strong> {q.q_a_similarity ?? "N/A"}</p>
                <p><strong>Answer length:</strong> {q.answer_length ?? 0}</p>
              </div>
            </>
          )}

          {/* Generic fallback for other question types */}
          {!(type === "role" || type === "resume" || type === "follow-up" || type === "followup") && (
            <div className="mt-2">
              <p className="text-sm text-gray-600">No specialized evaluation available for this question type.</p>
              <p className="mt-2"><strong>Q/A similarity:</strong> {q.q_a_similarity ?? "N/A"}</p>
              <p><strong>Answer length:</strong> {q.answer_length ?? 0}</p>
            </div>
          )}
        </div>
      );
    })
  ) : (
    <div className="text-gray-600">No per-question analysis available.</div>
  )}
</div>
    </div>
  );
}