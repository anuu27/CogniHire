import React, { useEffect, useRef, useState } from "react";
import CheatingDetection, {
  loadModels,
  detectMultipleFaces,
  ensureFacePresentAndForward,
  monitorTabSwitch,
} from "./CheatingDetection";
 
// Single-file updated VirtualInterviewerStarter component
// Key changes implemented:
// - RECORD_START_TIMEOUT_MS = 20s, treated as malpractice but does NOT skip the question
// - Detectors (face / tab) only run while an interview is active (session exists)
// - applyMalpractice is guarded to not queue/send events if no active session
// - All timers/intervals/cleanup handlers are cleared on interview end to avoid post-end logging
// - Malpractice queue flush tries to attach sessionIdRef when available and re-queues on failures

export default function VirtualInterviewerStarter() {
  // ---------------------- basic state & refs ----------------------
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeParsed, setResumeParsed] = useState(null);
  const [role, setRole] = useState("");
  const [cameraAllowed, setCameraAllowed] = useState(false);
  const [micAllowed, setMicAllowed] = useState(false);
  const [stream, setStream] = useState(null);
  const videoRef = useRef(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const [recording, setRecording] = useState(false);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");

  // roles and resume server name
  const [rolesList, setRolesList] = useState([]);
  const [selectedRole, setSelectedRole] = useState("");
  const [resumeServerName, setResumeServerName] = useState(null);

  // scoring & malpractice counters
  const [score, setScore] = useState(100); // start with 100 points
  const [malpracticeCount, setMalpracticeCount] = useState(0);

  // ---------- detection reliability refs ----------
  const modelsLoadedRef = useRef(false);
  const consecutiveNoFaceRef = useRef(0);
  const consecutiveMultiFaceRef = useRef(0);
  const lastFaceStateRef = useRef("present"); // "present" | "no-face" | "multi-face"

  // session ref to avoid stale closures
  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    sessionIdRef.current = sessionId;
    if (sessionId) flushMalpracticeQueueWithSession(sessionId);
  }, [sessionId]);

  // ---------- timers for questions & auto-end ----------
  const questionTimerRef = useRef(null); // expects startRecording within X seconds
  const autoEndTimerRef = useRef(null); // auto-end interview after total duration
  const questionStartedAtRef = useRef(null);
  const recordingStartedForQuestionRef = useRef(false);

  // Detector control refs
  const detectionIntervalRef = useRef(null);
  const tabSwitchCleanupRef = useRef(null);

  // configurable penalties / durations
  const RECORD_START_TIMEOUT_MS = 20 * 1000; // 20 seconds
  const INTERVIEW_MAX_MS = 30 * 60 * 1000; // 30 minutes auto-end
  const PENALTY_POINTS = 5; // points deducted per malpractice event

  // ---------- malpractice queueing helpers ----------
  const malpracticeQueueRef = useRef([]);
  const malpracticeTimerRef = useRef(null);
  const identityBaselineRef = useRef(null);           // face descriptor vector at start
const identityModelReadyRef = useRef(false);        // whether face embedding function available
const audioCtxRef = useRef(null);
const analyserRef = useRef(null);
const frequencyDataRef = useRef(null);
const timeDomainRef = useRef(null);
const candidateAudioProfileRef = useRef(null);      // spectral centroid average
const audioProfileBuiltRef = useRef(false);
const audioCheckIntervalRef = useRef(null);

// --- Configurable thresholds (tuneable)
const IDENTITY_SIMILARITY_WARN = 0.75; // similarity < this => warn
const IDENTITY_SIMILARITY_FLAG = 0.60; // similarity < this => malpractice
const AUDIO_SPECTRAL_DIFF_RATIO = 0.30; // relative change vs baseline -> flag
const AUDIO_RMS_THRESHOLD = 0.01;       // minimal RMS to consider speech present
const AUDIO_CHECK_INTERVAL_MS = 500;    // how often audio checks run
const IDENTITY_CHECK_INTERVAL_MS = 1500;// identity check rate


  // ---------- predefined roles ----------
  useEffect(() => {
    const predefinedRoles = [
      "AI Engineer",
      "AI Researcher",
      "ARVR_Developer",
      "Blockchain_Developer",
      "Business_Analyst",
      "Cloud_architect",
      "Cloud_engineer",
      "Content_writer",
      "Cybersecurity_Analyst",
      "Cybersecurity_specialist",
      "Data_analyst",
      "Data_architect",
      "Data_engineer",
      "Data_scientist",
      "Database_administrator",
      "Devops_engineer",
      "Digital_Marketing_Specialist",
      "E-commerce_Specialist",
      "Full_stack_Developer",
      "Game_Developer",
      "Graphic_Designer",
      "Human_Resources_Specialist",
      "IT_Support_Specialist",
      "Machine_Learning_Engineer",
      "Mobile_app_Developer",
      "Network_engineer",
      "Product_Manager",
      "Project_Manager",
      "QA_engineer",
      "Robotics_Engineer",
      "Software_Developer",
      "Software_engineer",
      "System_Administrator",
      "UI_designer",
      "UI_engineer",
      "UIUX_designer",
      "UX_designer",
    ];
    setRolesList(predefinedRoles);
    setSelectedRole(predefinedRoles[0]);
  }, []);

  // Load roles from backend if available
  useEffect(() => {
    let mounted = true;
    async function loadRoles() {
      try {
        const res = await fetch("/api/roles");
        if (!res.ok) throw new Error("Failed to fetch roles");
        const data = await res.json();
        const roles = data.roles || [];
        if (mounted && roles.length > 0) {
          setRolesList(roles);
          setSelectedRole(roles[0]);
        }
      } catch (err) {
        console.warn("Could not load roles (not fatal):", err);
      }
    }
    loadRoles();
    return () => { mounted = false; };
  }, []);

  // ---------------------- face models loading ----------------------
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await loadModels();
        if (mounted) {
          modelsLoadedRef.current = true;
          console.log("face models loaded");
        }
      } catch (err) {
        modelsLoadedRef.current = false;
        console.warn("Failed to load face detection models:", err);
      }
    })();
    return () => { mounted = false; };
  }, []);

  // -------------------- detectors (start only when interview active) --------------------
  useEffect(() => {
    // only run detectors when interview is active and we have a session
    if (!interviewStarted || !sessionIdRef.current) return;

    // Face detection interval (robust: require consecutive frames)
    const NO_FACE_THRESHOLD = 3;
    const MULTI_FACE_THRESHOLD = 2;

    detectionIntervalRef.current = setInterval(async () => {
      try {
        // after normal face checks (present / multi-face)
      
        const video = videoRef.current;
        if (!video || video.readyState < 2 || video.videoWidth === 0) return;

        const count = await detectMultipleFaces?.(video);
        if (typeof count !== "number") return;

        if (count === 0) {
          consecutiveNoFaceRef.current += 1;
          consecutiveMultiFaceRef.current = 0;
          if (consecutiveNoFaceRef.current >= NO_FACE_THRESHOLD && lastFaceStateRef.current !== "no-face") {
            lastFaceStateRef.current = "no-face";
            setStatusMsg("⚠ No face detected");
            // guard: ensure session active
            if (sessionIdRef.current && interviewStarted) {
              applyMalpractice({ type: "no_face_detected", details: { raw: "detector returned 0 faces" } });
            }
          }
        } else if (count > 1) {
          consecutiveMultiFaceRef.current += 1;
          consecutiveNoFaceRef.current = 0;
          if (consecutiveMultiFaceRef.current >= MULTI_FACE_THRESHOLD && lastFaceStateRef.current !== "multi-face") {
            lastFaceStateRef.current = "multi-face";
            setStatusMsg("❌ Multiple faces detected!");
            if (sessionIdRef.current && interviewStarted) {
              applyMalpractice({ type: "multiple_faces_detected", details: { raw: `detector returned ${count} faces` } });
            }
          }
        } else {
          lastFaceStateRef.current = "present";
          consecutiveNoFaceRef.current = 0;
          consecutiveMultiFaceRef.current = 0;
          setStatusMsg("");
        }
      } catch (e) {
        console.warn("detectMultipleFaces error:", e);
      }
    }, 1100);

    // Tab switch monitoring — only while interview active
    try {
      const cleanup = monitorTabSwitch?.((msg) => {
        if (msg && sessionIdRef.current && interviewStarted) {
          applyMalpractice({ type: "tab_switch", details: { raw: msg } });
          setStatusMsg((s) => `⚠ Cheating detected: ${msg}`);
        }
      });
      tabSwitchCleanupRef.current = typeof cleanup === "function" ? cleanup : null;
    } catch (e) {
      console.warn("monitorTabSwitch error:", e);
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      if (tabSwitchCleanupRef.current) {
        try { tabSwitchCleanupRef.current(); } catch (e) {}
        tabSwitchCleanupRef.current = null;
      }
    };
  }, [interviewStarted]);

  // ---------- ensure face present callback (also only runs while interview active) ----------
  useEffect(() => {
    if (!interviewStarted || !sessionIdRef.current) return;
    let interval = null;
    interval = setInterval(() => {
      try {
        if (!videoRef.current) return;
        const maybe = ensureFacePresentAndForward?.(videoRef.current, (warn) => {
          if (warn && sessionIdRef.current && interviewStarted) {
            const type = warn.toLowerCase().includes("face") ? "no_face_detected" : "gaze_warning";
            applyMalpractice({ type, details: { raw: warn } });
            setStatusMsg("⚠ " + warn);
          }
        });
        if (maybe && typeof maybe.then === "function") maybe.catch((err) => console.warn("ensureFacePresentAndForward error:", err));
      } catch (e) {
        console.warn("ensureFacePresentAndForward error:", e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [interviewStarted]);

  // ---------- resume upload ----------
  async function handleResumeUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setResumeFile(file);
    const form = new FormData();
    form.append("resume", file);

    try {
      const res = await fetch("/api/upload_resume", { method: "POST", body: form });
      const data = await res.json();
      setResumeParsed(data.parsed || null);

      if (data.saved_filename) setResumeServerName(data.saved_filename);
      else if (data.savedFilename) setResumeServerName(data.savedFilename);
      else if (data.parsed?.filename) setResumeServerName(data.parsed.filename);
      else setResumeServerName(null);

      setStatusMsg("Resume uploaded and parsed.");
      console.log("upload_resume response:", data);
    } catch (err) {
      console.error(err);
      setStatusMsg("Resume upload failed.");
    }
  }

  // ---------- media (camera + mic) ----------
  async function requestMedia() {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: true });
      setStream(s);
      setCameraAllowed(true);
      setMicAllowed(true);
      if (videoRef.current) videoRef.current.srcObject = s;
      setStatusMsg("Camera & mic allowed.");

      // audio analyser setup (non-blocking). call it here so analyserRef is available early.
      try {
        setupAudioAnalyser(s);
      } catch (e) {
        console.warn("setupAudioAnalyser failed:", e);
      }
    } catch (err) {
      console.error("getUserMedia failed:", err);
      setStatusMsg("Please allow camera/microphone permission (or use secure context).");
    }
  }
  useEffect(() => {
    return () => {
      // cleanup on unmount
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        try { mediaRecorderRef.current.stop(); } catch (e) {}
      }
      if (malpracticeTimerRef.current) clearTimeout(malpracticeTimerRef.current);
      if (questionTimerRef.current) clearTimeout(questionTimerRef.current);
      if (autoEndTimerRef.current) clearTimeout(autoEndTimerRef.current);
      if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);
      if (tabSwitchCleanupRef.current) { try { tabSwitchCleanupRef.current(); } catch (e) {} }
      
    };
  }, [stream]);

  // ---------- interview lifecycle ----------
  async function startInterview() {
  if (!resumeFile && !resumeServerName) return setStatusMsg("Upload resume first.");
  if (!cameraAllowed || !micAllowed) {
    setStatusMsg("Requesting camera & mic permissions...");
    await requestMedia();
    if (!cameraAllowed || !micAllowed) return setStatusMsg("Camera and mic required.");
  }

  const form = new FormData();
  if (resumeServerName) form.append("resume_path", resumeServerName);
  else form.append("resume_path", resumeFile?.name || "");

  const roleToSend = selectedRole || role || "Unknown";
  form.append("role", roleToSend);

  try {
    const res = await fetch("/api/start_interview", { method: "POST", body: form });
    const data = await res.json();
    console.log("startInterview response:", data, "sent role:", roleToSend, "sent resume:", resumeServerName || resumeFile?.name);

    if (data.session_id) {
      // set session first so detectors effect sees it
      setSessionId(String(data.session_id).trim());
    }

    // Wait a bit and attempt to capture identity baseline — but ensure video is ready
    setTimeout(async () => {
      try {
        const waitUntilVideoReady = async (timeoutMs = 3000) => {
          const start = Date.now();
          while (Date.now() - start < timeoutMs) {
            const v = videoRef.current;
            if (v && v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0) return true;
            // small wait
            // eslint-disable-next-line no-await-in-loop
            await new Promise(r => setTimeout(r, 200));
          }
          return false;
        };

        const ready = await waitUntilVideoReady(4000); // up to 4s for camera warm-up
        if (ready) {
          let ok = await captureIdentityBaseline(videoRef.current);
          if (!ok) {
            // try a couple of times with short delays (camera stabilization)
            await new Promise(r => setTimeout(r, 700));
            ok = await captureIdentityBaseline(videoRef.current);
            if (!ok) console.warn("captureIdentityBaseline: second attempt failed");
          }
        } else {
          // still not ready — try a single delayed attempt
          await new Promise(r => setTimeout(r, 800));
          try { await captureIdentityBaseline(videoRef.current); } catch (e) { console.warn("baseline capture delayed attempt failed", e); }
        }
      } catch (e) { console.warn("baseline capture failed", e); }
    }, 1200); // small delay to let camera warm-up

    // Build short audio profile in background (non-blocking).
    // If analyserRef isn't ready yet, retry a few times spaced out.
    (async function tryBuildAudioProfile(retries = 4, delayMs = 400) {
      for (let i = 0; i < retries; i++) {
        if (analyserRef.current) {
          try {
            await buildCandidateAudioProfile(3000).catch((e) => { throw e; });
            break;
          } catch (e) {
            console.warn("buildCandidateAudioProfile attempt failed:", e);
            // eslint-disable-next-line no-await-in-loop
            await new Promise(r => setTimeout(r, delayMs));
          }
        } else {
          // wait then try again
          // eslint-disable-next-line no-await-in-loop
          await new Promise(r => setTimeout(r, delayMs));
        }
      }
    })();

    if (data.first_question) {
      setCurrentQuestion(data.first_question);
      // start the question timer (candidate must start recording within configured time)
      startQuestionTimer();
      if (data.first_question.question && typeof window !== "undefined" && window.speechSynthesis) {
        try { speechSynthesis.speak(new SpeechSynthesisUtterance(data.first_question.question)); } catch (e) { console.warn("TTS failed:", e); }
      }
    }

    // setup auto-end timer for whole interview
    if (autoEndTimerRef.current) clearTimeout(autoEndTimerRef.current);
    autoEndTimerRef.current = setTimeout(() => {
      console.log("Auto-ending interview after max duration");
      // only apply malpractice if session active
      if (sessionIdRef.current && interviewStarted) applyMalpractice({ type: "auto_end_time_reached", details: { reason: "time_limit" } });
      endInterview();
    }, INTERVIEW_MAX_MS);

    // set interview started AFTER session id is set
    setInterviewStarted(true);
    setStatusMsg("Interview started.");
  } catch (err) {
    console.error(err);
    setStatusMsg("Could not start interview.");
  }
}
  // ---------- recording answers ----------
  function startRecording() {
    if (!stream) return setStatusMsg("Camera stream not available.");
    // Clear question timer because candidate began recording
    if (questionTimerRef.current) {
      clearTimeout(questionTimerRef.current);
      questionTimerRef.current = null;
    }
    recordingStartedForQuestionRef.current = true;

    chunksRef.current = [];
    let mime = "video/webm;codecs=vp8,opus";
    if (!MediaRecorder.isTypeSupported(mime)) {
      mime = "video/webm";
      if (!MediaRecorder.isTypeSupported(mime)) mime = "";
    }

    try {
      const mr = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => { const blob = new Blob(chunksRef.current, { type: "video/webm" }); await uploadAnswer(blob); };
      mr.start();
      setRecording(true);
      setStatusMsg("Recording...");
    } catch (err) {
      console.error("MediaRecorder start failed:", err);
      setStatusMsg("Recording failed (MediaRecorder not supported or wrong context).");
    }
  }

  function stopRecording() {
    if (!mediaRecorderRef.current) return;
    try { if (mediaRecorderRef.current.state !== "inactive") mediaRecorderRef.current.stop(); } catch (err) { console.warn("stopRecording error:", err); }
    setRecording(false);
    recordingStartedForQuestionRef.current = false;
    setStatusMsg("Stopped recording.");
  }

  // ---------- upload answer ----------
  async function uploadAnswer(blob, skipped = false) {
    if (!sessionIdRef.current) { setStatusMsg("No session id; cannot upload answer."); return; }
    const form = new FormData();
    form.append("session_id", sessionIdRef.current);
    if (skipped) form.append("skipped", "true");
    else form.append("answer_blob", blob, "answer.webm");
    form.append("question", currentQuestion ? currentQuestion.question : "");
    try {
      const res = await fetch("/api/process_answer", { method: "POST", body: form });
      const data = await res.json();

      if (data.next_question) {
        setCurrentQuestion(data.next_question);
        // reset recording flag
        recordingStartedForQuestionRef.current = false;
        // start timer for next question
        startQuestionTimer();
        if (data.next_question.question && typeof window !== "undefined" && window.speechSynthesis) {
          try { speechSynthesis.speak(new SpeechSynthesisUtterance(data.next_question.question)); } catch (e) { console.warn("TTS failed:", e); }
        }
      } else {
        setCurrentQuestion({ question: "Interview Completed!" });
        setInterviewStarted(false);
      }
      setStatusMsg("Answer uploaded.");
    } catch (err) {
      console.error(err);
      setStatusMsg("Failed to upload answer.");
      if (sessionIdRef.current && interviewStarted) applyMalpractice({ type: "upload_failed", details: { message: err?.message || String(err) } });
    }
  }

  // ---------- end interview ----------
  async function endInterview() {
    // don't confirm auto-end
    if (interviewStarted) {
      if (!window.confirm("Are you sure you want to end the interview? This will finalize and save your transcript.")) return;
    }

    try {
      // Stop detectors + timers immediately
      if (detectionIntervalRef.current) { clearInterval(detectionIntervalRef.current); detectionIntervalRef.current = null; }
      if (tabSwitchCleanupRef.current) { try { tabSwitchCleanupRef.current(); } catch (e) {} tabSwitchCleanupRef.current = null; }
      if (questionTimerRef.current) { clearTimeout(questionTimerRef.current); questionTimerRef.current = null; }
      if (autoEndTimerRef.current) { clearTimeout(autoEndTimerRef.current); autoEndTimerRef.current = null; }
      if (malpracticeTimerRef.current) { clearTimeout(malpracticeTimerRef.current); malpracticeTimerRef.current = null; }
      cleanupAudioAnalyser();
      identityBaselineRef.current = null;
      identityModelReadyRef.current = false;

      if (recording) {
        stopRecording();
        // let recorder finalize
        await new Promise((res) => setTimeout(res, 800));
      }

      if (!sessionIdRef.current) {
        setStatusMsg("No active session to end.");
        // also clear client state
        setInterviewStarted(false);
        setSessionId(null);
        return;
      }

      setStatusMsg("Ending interview and saving transcript...");
      const form = new FormData(); form.append("session_id", sessionIdRef.current);
      form.append("final_score", String(score));
      const res = await fetch("/api/end_interview", { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setInterviewStarted(false);
        setCurrentQuestion({ question: "Interview ended.", type: "end" });
        setStatusMsg("Interview ended. Transcript saved.");
        console.log("Transcript saved to:", data.transcript_path);
      } else {
        console.error("End interview failed:", data);
        setStatusMsg("Failed to end interview: " + (data.error || JSON.stringify(data)));
      }
    } catch (err) {
      console.error("endInterview error:", err);
      setStatusMsg("Network error ending interview: " + (err && err.message));
    } finally {
      // ensure we clear sessionId immediately to prevent post-end logging from queued flushes
      sessionIdRef.current = null;
      setSessionId(null);
      setInterviewStarted(false);
      resetToInitial();
    }
  }

  // ---------------------- malpractice helpers ----------------------
  async function sendMalpracticeToServer(envelope) {
    try {
      const rawSid = envelope.session_id || sessionIdRef.current || "";
      const sid = rawSid ? String(rawSid).trim() : "";
      const body = { session_id: sid || null, event: envelope.event || envelope };

      console.log("DEBUG sendMalpracticeToServer -> sending:", body);

      if (!body.session_id) {
        console.warn("sendMalpracticeToServer: no session_id, queueing instead", body);
        malpracticeQueueRef.current.push(body);
        return { ok: false, error: "no_session_id" };
      }

      const res = await fetch("/api/log_malpractice", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body),
      });

      const text = await res.text().catch(() => null);
      let parsed = null;
      try { parsed = text ? JSON.parse(text) : null; } catch (e) {}

      if (!res.ok) {
        console.warn("malpractice log failed:", res.status, res.statusText, text, parsed);
        malpracticeQueueRef.current.push(body);
        return { ok: false, status: res.status, body: parsed || text };
      } else {
        console.log("malpractice logged OK:", body.event.type, parsed || text);
        return { ok: true, status: res.status, body: parsed || text };
      }
    } catch (err) {
      console.error("malpractice logging error (network):", err);
      malpracticeQueueRef.current.push(envelope);
      return { ok: false, error: String(err) };
    }
  }

  function logMalpracticeDebounced(payload) {
    const event = payload.event || { type: payload.type || payload.event_type || "unknown", timestamp: payload.timestamp || new Date().toISOString(), details: payload.details || {} };
    const envelope = { session_id: sessionIdRef.current || null, event };

    // If no active session, queue it for flush when session becomes available
    if (!envelope.session_id) {
      malpracticeQueueRef.current.push(envelope);
      console.log("Queued malpractice (no session yet):", envelope);
      return;
    }

    malpracticeQueueRef.current.push(envelope);

    if (malpracticeTimerRef.current) clearTimeout(malpracticeTimerRef.current);
    malpracticeTimerRef.current = setTimeout(async () => {
      const batch = malpracticeQueueRef.current.splice(0);
      console.log("Flushing malpractice batch:", batch);
      for (const item of batch) {
        if (!item.session_id) item.session_id = sessionIdRef.current || null;
        await sendMalpracticeToServer(item);
      }
      malpracticeTimerRef.current = null;
    }, 800);
  }

  async function flushMalpracticeQueueWithSession(newSessionId) {
    if (!newSessionId) return;
    if (!malpracticeQueueRef.current.length) return;
    console.log("Flushing queued malpractice events with session:", newSessionId);
    const queued = malpracticeQueueRef.current.splice(0);
    for (const item of queued) {
      const toSend = { session_id: newSessionId, event: item.event || item };
      await sendMalpracticeToServer(toSend);
    }
  }

  // centralized helper to apply malpractice locally and send to server
  function applyMalpractice(payload) {
    // Guard: only apply if interview active and session present
    if (!sessionIdRef.current || !interviewStarted) {
      console.log("Dropping malpractice (no active session):", payload);
      return;
    }

    // increment local count and reduce score
    setMalpracticeCount((c) => c + 1);
    setScore((s) => Math.max(0, s - PENALTY_POINTS));

    // include current question context if available
    const eventPayload = {
      event: {
        type: payload.type || "unknown",
        timestamp: new Date().toISOString(),
        details: { ...(payload.details || {}), question: currentQuestion?.question || null }
      },
      session_id: sessionIdRef.current || null,
    };

    // queue and debounced send
    logMalpracticeDebounced(eventPayload);
  }

  // ---------------------- question timer management ----------------------
  function startQuestionTimer() {
    // clear any existing timer
    if (questionTimerRef.current) clearTimeout(questionTimerRef.current);
    recordingStartedForQuestionRef.current = false;
    questionStartedAtRef.current = Date.now();

    questionTimerRef.current = setTimeout(async () => {
      if (!recordingStartedForQuestionRef.current) {
        console.log("QUESTION TIMER -> candidate did not start recording within timeout. Flagging but DO NOT skip question.");
        applyMalpractice({ type: "no_record_start_within_timeout", details: { timeout_ms: RECORD_START_TIMEOUT_MS } });
        setStatusMsg("⚠ You did not start recording within 10 seconds — marked as malpractice. Please start recording to answer the same question.");
        // DO NOT skip or move to next question — candidate can still start recording and answer
      }
    }, RECORD_START_TIMEOUT_MS);
  }

  // ---------------------- small instrumentation ----------------------
  useEffect(() => { console.log("Session ID updated:", sessionId); }, [sessionId]);

  useEffect(() => {
    const t = setInterval(() => {
      if (malpracticeQueueRef.current && malpracticeQueueRef.current.length) {
        console.log("Malpractice queue size:", malpracticeQueueRef.current.length, "queue:", malpracticeQueueRef.current);
      }
    }, 3000);
    return () => clearInterval(t);
  }, []);
  /****************** START DROP-IN: Identity + Audio checks ******************/
  function cosineSimilarity(a = [], b = []) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0) return 0;
    return dot / Math.sqrt(na * nb);
  }

  // --- Helper: convert Mediapipe/face-api landmarks or descriptor to vector
  // If you use face-api.js it will give a 128-d descriptor; if you use Mediapipe FaceMesh you
  // should flatten landmarks into a vector. This function tries to handle both shapes.
  function normalizeFaceVector(raw) {
    if (!raw) return null;
    // If face-api descriptor (Float32Array length ~128)
    if (raw.length && typeof raw[0] === "number" && raw.length >= 64) {
      return Array.from(raw);
    }
    // If Mediapipe landmarks array [{x,y,z},...]
    if (Array.isArray(raw) && raw.length && raw[0].x !== undefined) {
      const v = [];
      for (const p of raw) { v.push(p.x, p.y, p.z); }
      // Normalize magnitude (optional)
      const mean = v.reduce((s,x)=>s+x,0)/v.length;
      const std = Math.sqrt(v.reduce((s,x)=>s + (x-mean)*(x-mean),0)/v.length) || 1;
      return v.map(x => (x-mean)/std);
    }
    return null;
  }

  // --- Face embedding accessor (tries multiple runtimes; must return vector or null)
  // This expects that one of these globals exists based on your current detection code:
  //  - window.faceapi with net descriptors, OR
  //  - a `detectFaceDescriptor(video)` utility that returns descriptor, OR
  //  - your existing detectMultipleFaces may be backed by face-api; we need descriptor
  async function getFaceDescriptorFromVideo(videoEl) {
    if (!videoEl) return null;

    // Option 1: face-api.js (common)
    try {
      if (window.faceapi && faceapi.detectSingleFace) {
        // ensure models loaded in your loadModels step
        const detection = await faceapi.detectSingleFace(videoEl).withFaceLandmarks().withFaceDescriptor();
        if (detection && detection.descriptor) return Array.from(detection.descriptor);
      }
    } catch (e) {
      // ignore and try next
    }

    // Option 2: if CheatingDetection exported a descriptor function (optional)
    try {
      if (typeof window.getFaceDescriptorFromVideo === "function") {
        const d = await window.getFaceDescriptorFromVideo(videoEl);
        if (d) return normalizeFaceVector(d);
      }
    } catch (e) {}

    // Option 3: If your detectMultipleFaces returns landmarks somewhere accessible:
    // (Only use this if you can expose landmarks from your detection)
    try {
      if (typeof window.getLastFaceLandmarks === "function") {
        const landmarks = await window.getLastFaceLandmarks();
        if (landmarks) return normalizeFaceVector(landmarks);
      }
    } catch (e) {}

    // No descriptor available
    return null;
  }

  // --- Capture baseline embedding after start (call once when face stable)
  async function captureIdentityBaseline(videoEl) {
    try {
      const desc = await getFaceDescriptorFromVideo(videoEl);
      if (!desc) return false;
      const vec = normalizeFaceVector(desc);
      if (!vec) return false;
      identityBaselineRef.current = vec;
      identityModelReadyRef.current = true;
      console.log("Captured identity baseline (len=" + vec.length + ")");
      return true;
    } catch (e) {
      console.warn("captureIdentityBaseline failed", e);
      return false;
    }
  }

  // --- Run identity check against baseline; call applyMalpractice on mismatch
  async function checkIdentityAgainstBaseline(videoEl) {
    if (!identityBaselineRef.current) return;
    try {
      const desc = await getFaceDescriptorFromVideo(videoEl);
      if (!desc) return; // no face; no identity check here (no-face handled by other logic)
      const vec = normalizeFaceVector(desc);
      if (!vec) return;
      // resize vectors to same length if possible
      const base = identityBaselineRef.current;
      const minLen = Math.min(base.length, vec.length);
      const s = cosineSimilarity(base.slice(0,minLen), vec.slice(0,minLen));
      // s in [-1,1] usually, but descriptors are positive-normalized; treat s ~ [0,1]
      if (s < IDENTITY_SIMILARITY_FLAG) {
        console.log("IDENTITY MISMATCH flagged: sim=", s);
        applyMalpractice({ type: "identity_mismatch", details: { similarity: s } });
        setStatusMsg(`⚠ Identity mismatch: ${Math.round(s*100)/100}`);
      } else if (s < IDENTITY_SIMILARITY_WARN) {
        // optional: a warning (no immediate penalty)
        console.log("IDENTITY low confidence (warn): sim=", s);
        setStatusMsg(`Identity confidence low (${Math.round(s*100)}%)`);
      } else {
        // OK: clear any small status if previously set
        // only clear if status message matches identity warnings (avoid clobbering others)
        // setStatusMsg("");
      }
    } catch (e) {
      console.warn("checkIdentityAgainstBaseline error", e);
    }
  }

  // --- AUDIO: set up analyser when stream is created
  function setupAudioAnalyser(stream) {
    try {
      if (!stream) return;
      if (!audioCtxRef.current) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        audioCtxRef.current = new AudioContextClass();
      }
      if (analyserRef.current) return;
      const ctx = audioCtxRef.current;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      frequencyDataRef.current = new Float32Array(analyser.frequencyBinCount);
      timeDomainRef.current = new Float32Array(analyser.fftSize);
      src.connect(analyser);
      // ensure audio analysis interval runs
      if (audioCheckIntervalRef.current) clearInterval(audioCheckIntervalRef.current);
      audioCheckIntervalRef.current = setInterval(() => {
        try { runAudioChecks(); } catch (e) { console.warn("audio check err", e); }
      }, AUDIO_CHECK_INTERVAL_MS);
      console.log("Audio analyser set up");
    } catch (e) {
      console.warn("setupAudioAnalyser error", e);
    }
  }

  // --- Helpers: RMS and spectral centroid
  function computeRMS() {
    const analyser = analyserRef.current;
    if (!analyser) return 0;
    analyser.getFloatTimeDomainData(timeDomainRef.current);
    let sum = 0;
    for (let i = 0; i < timeDomainRef.current.length; i++) sum += timeDomainRef.current[i] * timeDomainRef.current[i];
    return Math.sqrt(sum / timeDomainRef.current.length);
  }

  function computeSpectralCentroid() {
    const analyser = analyserRef.current;
    if (!analyser) return 0;
    analyser.getFloatFrequencyData(frequencyDataRef.current); // decibel values
    let num = 0, den = 0;
    const binCount = frequencyDataRef.current.length;
    const nyquist = (audioCtxRef.current?.sampleRate || 44100) / 2;
    for (let i = 0; i < binCount; i++) {
      const magDb = frequencyDataRef.current[i];
      // convert to linear magnitude
      const mag = Math.pow(10, magDb / 20);
      const freq = (i / binCount) * nyquist;
      num += freq * mag;
      den += mag;
    }
    return den ? (num / den) : 0;
  }

  // --- Build candidate audio profile (call once at start, e.g. during first question)
  async function buildCandidateAudioProfile(durationMs = 3000) {
    if (!analyserRef.current) return false;
    const end = Date.now() + durationMs;
    const vals = [];
    while (Date.now() < end) {
      const rms = computeRMS();
      if (rms > AUDIO_RMS_THRESHOLD) {
        const sc = computeSpectralCentroid();
        if (sc) vals.push(sc);
      }
      // small sleep
      // eslint-disable-next-line no-await-in-loop
      await new Promise(res => setTimeout(res, 200));
    }
    if (vals.length) {
      const avg = vals.reduce((a,b)=>a+b,0)/vals.length;
      candidateAudioProfileRef.current = avg;
      audioProfileBuiltRef.current = true;
      console.log("Built audio profile, centroid=", avg);
      return true;
    }
    return false;
  }

  // --- periodic audio checks
  function runAudioChecks() {
    if (!analyserRef.current || !audioProfileBuiltRef.current) return;
    const rms = computeRMS();
    const sc = computeSpectralCentroid();
    const baseline = candidateAudioProfileRef.current || 0;
    if (rms > AUDIO_RMS_THRESHOLD) {
      // if speech is present and spectral centroid differs a lot, suspect other voice/source
      if (baseline > 0) {
        const diffRatio = Math.abs(sc - baseline) / (baseline + 1e-9);
        if (diffRatio > AUDIO_SPECTRAL_DIFF_RATIO) {
          console.log("AUDIO: spectral diff flagged", diffRatio, sc, baseline);
          applyMalpractice({ type: "background_voice_detected", details: { spectral_centroid: sc, diff_ratio: diffRatio } });
          setStatusMsg("⚠ Background voice detected");
        }
      }
    }
  }

  // --- Cleanup audio resources
  function cleanupAudioAnalyser() {
    try {
      if (audioCheckIntervalRef.current) { clearInterval(audioCheckIntervalRef.current); audioCheckIntervalRef.current = null; }
      if (analyserRef.current) { try { analyserRef.current.disconnect(); } catch (e) {} analyserRef.current = null; }
      if (audioCtxRef.current) { try { audioCtxRef.current.close(); } catch (e) {} audioCtxRef.current = null; }
      candidateAudioProfileRef.current = null; audioProfileBuiltRef.current = false;
    } catch (e) {
      console.warn("cleanupAudioAnalyser error", e);
    }
  }
  // ------ add these helpers (place with other helper functions) ------

function clearAllDetectorsAndTimers() {
  if (detectionIntervalRef.current) { clearInterval(detectionIntervalRef.current); detectionIntervalRef.current = null; }
  if (tabSwitchCleanupRef.current) { try { tabSwitchCleanupRef.current(); } catch(e){} tabSwitchCleanupRef.current = null; }
  if (questionTimerRef.current) { clearTimeout(questionTimerRef.current); questionTimerRef.current = null; }
  if (autoEndTimerRef.current) { clearTimeout(autoEndTimerRef.current); autoEndTimerRef.current = null; }
  if (malpracticeTimerRef.current) { clearTimeout(malpracticeTimerRef.current); malpracticeTimerRef.current = null; }
}

function stopAndReleaseMedia() {
  try {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      try { mediaRecorderRef.current.stop(); } catch(e){}
    }
  } catch(e){}

  try {
    if (stream) { stream.getTracks().forEach(t => { try { t.stop(); } catch(e){} }); }
  } catch(e){}

  try { if (videoRef.current) videoRef.current.srcObject = null; } catch(e){}
  setStream(null);
  setCameraAllowed(false);
  setMicAllowed(false);
}

async function flushAndClearMalpracticeQueue() {
  try {
    if (malpracticeQueueRef.current && malpracticeQueueRef.current.length) {
      const queued = malpracticeQueueRef.current.splice(0);
      for (const item of queued) {
        try {
          const body = { session_id: sessionIdRef.current || item.session_id || null, event: item.event || item };
          await fetch("/api/log_malpractice", { method: "POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify(body) });
        } catch(e) { /* ignore flush errors */ }
      }
    }
  } catch(e) {}
  malpracticeQueueRef.current = [];
}

function resetToInitial() {
  // stop detectors/timers and media
  clearAllDetectorsAndTimers();
  stopAndReleaseMedia();

  // cancel any TTS
  try { if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel(); } catch(e){}

  // reset react state
  setRecording(false);
  setInterviewStarted(false);
  setCurrentQuestion(null);
  setStatusMsg("");
  setSessionId(null);
  sessionIdRef.current = null;

  // reset counters/refs
  setScore(100);
  setMalpracticeCount(0);
  questionStartedAtRef.current = null;
  recordingStartedForQuestionRef.current = false;
  consecutiveNoFaceRef.current = 0;
  consecutiveMultiFaceRef.current = 0;
  lastFaceStateRef.current = "present";

  // flush or clear queued malpractice events
  flushAndClearMalpracticeQueue().catch(()=>{ malpracticeQueueRef.current = []; });
}

  /****************** END DROP-IN: Identity + Audio checks ******************/


  // ---------------------- JSX UI ----------------------
  return (
    
      <div
  className="min-h-screen bg-cover bg-center p-6"
  style={{ backgroundImage: "url('/bg3.jpg')" }}
>

      
        {/* BRAND HEADER (pure UI — no logic changes) */}
<div className="max-w-3xl mx-auto bg-white p-6 rounded-xl shadow">
        

<img src="/logo6.png" className="bg-transparent" />



<div className="mb-6">
  <div className="rounded-xl bg-gradient-to-r from-[#FFFFFF] via-[#b862a8] to-[#6d1371] text-white p-4 shadow-2xl flex items-center gap-4">
    
    <div className="max-w-3xl mx-auto bg-white p-6 rounded-xl shadow">
        <h1 className="text-2xl font-semibold mb-4">AI Virtual Interviewer</h1>

      
    </div>

    {/* spacer */}
    <div className="flex-1" />

    
    
  </div>
</div>
      

        
    
        {/* Resume Upload */}
        <label className="font-medium mt-4 block">Upload Resume</label>
        <input type="file" accept=".pdf,.doc,.docx" onChange={handleResumeUpload} />
        {resumeParsed && (
          <pre className="bg-gray-100 p-2 rounded mt-6 text-sm">{JSON.stringify(resumeParsed, null, 2)}</pre>
        )}

        {/* Role Selection */}
        <div className="mt-4">
          <label className="block font-medium">Role</label>

          {rolesList.length > 0 ? (
            <select value={selectedRole} onChange={(e) => setSelectedRole(e.target.value)} className="bg-[#d9dcde] border p-2 rounded w-full mt-1">
              {rolesList.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          ) : (
            <input className="border p-2 rounded w-full mt-1" value={role} onChange={(e) => setRole(e.target.value)} placeholder="e.g. Backend Engineer" />
          )}

          <div className="text-xl text-gray-500 mt-1">{rolesList.length > 0 ? "Choose a role from dataset" : "No roles loaded — type a role"}</div>
        </div>
        {/* Video Preview & Media Request */}
        <div className="mt-4">
          <div
            className="rounded-lg overflow-hidden border bg-black"
            style={{
              width: "300px",      // <<< CHANGE THIS TO ANY SIZE YOU WANT
              height: "200px",     // <<< ADJUST HEIGHT
              position: "relative",
            }}
          >
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover", // crop nicely
              }}
            />
          </div>

          <button
            onClick={requestMedia}
            className="bg-blue-600 text-white px-4 py-2 mt-3 rounded"
          >
            Enable Camera & Mic
          </button>
        </div>


        {/* Start Interview */}
        <button
  onClick={startInterview}
  disabled={interviewStarted}
  className="
    bg-[#b862a8]
    text-white
    font-bold
    text-3xl
    rounded-2xl
    shadow-xl
    hover:bg-[#723a6e]
    transition
    disabled:opacity-50
    disabled:cursor-not-allowed
    block
    w-[300px]     /* Width made large */
    h-[50px]      /* Height made large */
  text-xl"
>
  {interviewStarted ? "Interview in progress" : "Start Interview"}
</button>


        {/* Interview Question Section */}
        {interviewStarted && currentQuestion && (
          <div className="mt-6 bg-gray-50 p-4 rounded shadow">
            <h2 className="text-lg font-large">Question:</h2>
            <p className="mt-2">{currentQuestion.question}</p>

            <div className="mt-4 flex gap-3">
              <button onClick={startRecording} className="bg-red-600 text-white px-4 py-2 rounded" disabled={recording}>Start Answer</button>
              <button onClick={stopRecording} className="bg-gray-700 text-white px-4 py-2 rounded" disabled={!recording}>Stop Answer</button>
              <button onClick={endInterview} className="bg-indigo-600 text-white px-4 py-2 rounded ml-2">End Interview</button>
            </div>

            <div className="mt-3 text-sm text-gray-600">You must start recording within 10 seconds after the question is read. Otherwise it's flagged as malpractice but you can still answer.</div>
          </div>
        )}

        <p className="mt-4 text-sm text-gray-500">{statusMsg}</p>
      </div>
    </div>
  
  );
}
