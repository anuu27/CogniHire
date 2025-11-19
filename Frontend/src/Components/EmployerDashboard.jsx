// EmployerDashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import ReportDetail from "./ReportDetail";
import { motion } from "framer-motion";
import { Search } from "lucide-react";

function DashboardHeader({ total = 0, avg = 0, onSearch }) {
  return (
    <header className="mb-6">
      <div className="rounded-2xl bg-gradient-to-r from-brand-700 to-accent-500 text-white p-6 shadow-xl">
        <div className="flex items-center justify-between gap-6">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold"> Employer Dashboard</h1>
            <p className="mt-1 text-sm opacity-90">Live interview analytics & reports</p>
            <div className="mt-4 flex gap-3">
              <div className="bg-white/10 px-3 py-1 rounded-lg text-sm">
                <div className="text-xs opacity-80">Total interviews</div>
                <div className="font-semibold">{total}</div>
              </div>
              <div className="bg-white/10 px-3 py-1 rounded-lg text-sm">
                <div className="text-xs opacity-80">Avg score</div>
                <div className="font-semibold">{Math.round(avg * 10) / 10} / 100</div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:block">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/80" size={16} />
                <input
                  onChange={(e) => onSearch && onSearch(e.target.value)}
                  placeholder="Search by name or session id..."
                  className="pl-10 pr-3 py-2 rounded-lg bg-white/10 placeholder-white/70 focus:outline-none focus:ring-2 focus:ring-white/40"
                />
              </div>
            </div>
           
          </div>
        </div>
      </div>
    </header>
  );
}

function ScorePill({ score }) {
  const s = Number(score) || 0;
  const color = s >= 85 ? "bg-green-600" : s >= 70 ? "bg-lime-600" : s >= 50 ? "bg-yellow-500" : "bg-red-500";
  return <span className={`text-white px-2 py-1 rounded text-sm ${color}`}>{Math.round(s)} / 100</span>;
}

export default function EmployerDashboard() {
  const [reports, setReports] = useState([]);
  const [selected, setSelected] = useState(null); // full report object or summary
  const [loading, setLoading] = useState(false);
  const [roleFilter, setRoleFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState("date"); // date | score

  // UI states
  const [showDetail, setShowDetail] = useState(false); // when true -> show ONLY report pane
  const [showInterviewWindow, setShowInterviewWindow] = useState(true); // interview UI visible when showDetail==false

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    setLoading(true);
    try {
      const res = await fetch("/api/employer/reports");
      const json = await res.json();
      setReports(json.reports || []);
    } catch (err) {
      console.error("Failed to fetch reports:", err);
      setReports([]);
    } finally {
      setLoading(false);
    }
  }

  // Open a report: fetch full report, show only report pane, and close interview UI
  async function openReport(sessionId) {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/employer/report/${sessionId}`);
      const data = await res.json();

      // setSelected only if we have data, otherwise fallback to summary if available
      if (res.ok && data) {
        setSelected(data);
      } else {
        const summary = reports.find((r) => r.session_id === sessionId) || { session_id: sessionId, error: data?.error || "Report not available" };
        setSelected(summary);
      }

      // CRITICAL: hide interview UI whenever a report is selected
      setShowInterviewWindow(false);
      setShowDetail(true);
    } catch (e) {
      console.error("Failed to fetch report:", e);
      const summary = reports.find((r) => r.session_id === sessionId) || { session_id: sessionId, error: "Network error" };
      setSelected(summary);
      setShowInterviewWindow(false);
      setShowDetail(true);
    } finally {
      setLoading(false);
    }
  }

  // Close report and go back to Interviews window (list + interview player)
  function closeReport() {
    setShowDetail(false);
    setSelected(null);
    setShowInterviewWindow(true);
  }

  const roles = useMemo(() => ["All", ...Array.from(new Set(reports.map((r) => r.role || "Unknown")))], [reports]);

  const filtered = useMemo(() => {
    const q = (search || "").trim().toLowerCase();
    let arr = reports.filter((r) => {
      if (roleFilter !== "All" && (r.role || "Unknown") !== roleFilter) return false;
      if (!q) return true;
      return (r.candidate_name || "").toLowerCase().includes(q) || (r.session_id || "").toLowerCase().includes(q);
    });

    if (sortKey === "score") {
      arr = arr.sort((a, b) => (b.overall_score || 0) - (a.overall_score || 0));
    } else {
      arr = arr.sort((a, b) => {
        const ta = new Date(a.timestamp || a.end_time || 0).getTime();
        const tb = new Date(b.timestamp || b.end_time || 0).getTime();
        return tb - ta;
      });
    }
    return arr;
  }, [reports, roleFilter, search, sortKey]);

  const avg = useMemo(() => {
    if (!reports.length) return 0;
    return reports.reduce((s, r) => s + (r.overall_score || 0), 0) / reports.length;
  }, [reports]);

  // --- RENDER ---

  // If a report is selected, render ONLY the report view (full width)
  if (showDetail && selected) {
    return (
      <div
  className="min-h-screen p-6 bg-cover bg-center bg-no-repeat"
  style={{ backgroundImage: "url('/bg2.jpg')" }}
>

        <DashboardHeader total={reports.length} avg={avg} onSearch={setSearch} />

        <main className="max-w-5xl mx-auto mt-6 bg-white p-6 rounded shadow">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold">Report: {selected.session_id || selected.candidate_name}</h2>
              <div className="text-sm text-gray-500">{selected.candidate_name}</div>
            </div>

            <div className="flex items-center gap-2">
              <button
                className="text-sm px-3 py-1 border rounded"
                onClick={() => {
                  // download JSON
                  const blob = new Blob([JSON.stringify(selected, null, 2)], { type: "application/json" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `${selected.session_id || "report"}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Download JSON
              </button>

              <button className="text-sm px-3 py-1 bg-red-50 text-red-600 border border-red-100 rounded" onClick={closeReport}>
                Back to Interviews
              </button>
            </div>
          </div>

          {/* ONLY render ReportDetail if selected is truthy */}
          <ReportDetail
            report={selected}
            onDownload={() => {
              const blob = new Blob([JSON.stringify(selected, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `${selected.session_id || "report"}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            showInterview={false}
          />
        </main>
      </div>
    );
  }

  // Default: show Interviews window (list + optional interview player)
  return (
    <div
  className="min-h-screen p-6 bg-cover bg-center bg-no-repeat"
  style={{ backgroundImage: "url('/bg2.jpg')" }}
>

      <DashboardHeader total={reports.length} avg={avg} onSearch={setSearch} />

      <div className="max-w-7xl mx-auto grid grid-cols-1 gap-6">
        {/* Interviews list */}
        <div className="bg-white p-4 rounded shadow">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Interviews</h2>
            <div className="flex items-center gap-2">
              <button className="text-sm text-blue-600" onClick={loadReports}>Refresh</button>
            </div>
          </div>

          <div className="mt-3 flex gap-2">
            <select className="border p-2 rounded" value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)}>
              {roles.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>

            <select className="border p-2 rounded" value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
              <option value="date">Newest</option>
              <option value="score">Top score</option>
            </select>

            <input
              className="ml-auto border p-2 rounded w-64"
              placeholder="Search name or session id"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="mt-3 text-sm text-gray-500">Total: {reports.length} • Showing: {filtered.length}</div>

          <div className="mt-3 overflow-y-auto" style={{ maxHeight: "60vh" }}>
            {loading && <div className="py-6 text-center text-gray-500">Loading...</div>}
            {!loading && filtered.map(r => (
              <div key={r.session_id} className="py-3 border-b last:border-b-0 flex items-center justify-between gap-4">
                <div>
                  <div className="font-medium">{r.candidate_name || r.session_id}</div>
                  <div className="text-xs text-gray-500">{r.role} • {r.timestamp ? new Date(r.timestamp).toLocaleString() : ""}</div>
                </div>

                <div className="flex items-center gap-3">
                  <ScorePill score={r.overall_score ?? 0} />
                  <button
                    onClick={() => openReport(r.session_id)}
                    className="text-sm bg-indigo-600 text-white px-3 py-1 rounded hover:bg-indigo-700"
                  >
                    View
                  </button>
                </div>
              </div>
            ))}

            {!loading && filtered.length === 0 && <div className="py-6 text-center text-gray-400">No interviews found</div>}
          </div>
        </div>

        {/* Interview window (only visible when not viewing a report) */}
        {showInterviewWindow && (
          <div className="bg-white p-4 rounded shadow">
            {/* Replace this block with your actual interview player/component */}
            <div className="text-gray-700">Interview window / player — visible only in the Interviews view.</div>
          </div>
        )}
      </div>
    </div>
  );
}
