import { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";
const POLL_INTERVAL = 2500;

function ProgressRing({ progress, size = 80, stroke = 6 }) {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (progress / 100) * circ;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1e293b" strokeWidth={stroke} />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#38bdf8" strokeWidth={stroke}
        strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 0.4s ease" }} />
    </svg>
  );
}

function ConfidenceBar({ label, value, rank }) {
  const colors = ["#38bdf8", "#818cf8", "#a78bfa", "#c084fc", "#e879f9"];
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "#cbd5e1", fontSize: 13, fontFamily: "monospace" }}>{label}</span>
        <span style={{ color: colors[rank], fontSize: 13, fontWeight: 700 }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: "#1e293b", borderRadius: 99, height: 6, overflow: "hidden" }}>
        <div style={{
          width: `${value * 100}%`, height: "100%",
          background: `linear-gradient(90deg, ${colors[rank]}, ${colors[rank]}99)`,
          borderRadius: 99, transition: "width 1s cubic-bezier(.4,0,.2,1)"
        }} />
      </div>
    </div>
  );
}

export default function App() {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [fakeProgress, setFakeProgress] = useState(0);
  const pollRef = useRef(null);
  const progressRef = useRef(null);
  const fileInputRef = useRef(null);

  const startFakeProgress = () => {
    setFakeProgress(0);
    let p = 0;
    progressRef.current = setInterval(() => {
      p = Math.min(p + Math.random() * 3, 88);
      setFakeProgress(p);
    }, 300);
  };

  const stopFakeProgress = (success) => {
    clearInterval(progressRef.current);
    setFakeProgress(success ? 100 : 0);
  };

  const startPolling = (id) => {
  let attempts = 0;
  pollRef.current = setInterval(async () => {
    attempts++;
    try {
      const res = await fetch(`http://localhost:8000/job/${id}`);
      const data = await res.json();
      console.log("Poll response:", data); // see in browser console
      if (data.status === "done") {
        clearInterval(pollRef.current);
        stopFakeProgress(true);
        setStatus("done");
        setResult(data.result);
      } else if (data.status === "error") {
        clearInterval(pollRef.current);
        stopFakeProgress(false);
        setStatus("error");
        setError(data.error || "Processing failed");
      } else if (attempts > 120) {
        clearInterval(pollRef.current);
        setStatus("error");
        setError("Timeout: processing took too long");
      }
    } catch (e) {
      console.error("Poll error:", e);
    }
  }, POLL_INTERVAL);
};
  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setStatus("idle");
    setResult(null);
    setError(null);
    setJobId(null);
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    setStatus("uploading");
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/upload-and-process`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
      const data = await res.json();
      setJobId(data.job_id);
      setStatus("processing");
      startFakeProgress();
      startPolling(data.job_id);
    } catch (e) {
      setStatus("error");
      setError(e.message);
    }
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = `${API}/download/${jobId}`;
    a.download = `enhanced_${file?.name || "video.mp4"}`;
    a.click();
  };

  const reset = () => {
    clearInterval(pollRef.current);
    clearInterval(progressRef.current);
    setFile(null); setVideoUrl(null); setJobId(null);
    setStatus("idle"); setResult(null); setError(null); setFakeProgress(0);
  };

  const statusLabel = {
    idle: "Ready",
    uploading: "Uploading…",
    processing: "Enhancing & recognizing…",
    done: "Complete!",
    error: "Error occurred",
  }[status];

  return (
    <div style={{
      minHeight: "100vh",
      width: "100vw",
      background: "#020617",
      fontFamily: "'Inter', sans-serif",
      color: "#f1f5f9",
      boxSizing: "border-box",
      overflowX: "hidden",
    }}>

      {/* ── Header ── */}
      <div style={{
        borderBottom: "1px solid #0f172a",
        background: "linear-gradient(180deg, #0f172a 0%, #020617 100%)",
        padding: "24px 40px",
        display: "flex", alignItems: "center", gap: 16,
        width: "100%", boxSizing: "border-box",
      }}>
        <div style={{
          width: 44, height: 44, borderRadius: 12, flexShrink: 0,
          background: "linear-gradient(135deg, #38bdf8, #818cf8)",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22,
        }}>🎬</div>
        <div>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: -0.5, color: "#f1f5f9" }}>
            Noise Resilient Video Action Recognition System
          </h1>
        </div>
        {status !== "idle" && (
          <button onClick={reset} style={{
            marginLeft: "auto", background: "transparent",
            border: "1px solid #334155", borderRadius: 8,
            color: "#94a3b8", cursor: "pointer", padding: "6px 16px", fontSize: 13,
          }}>Reset</button>
        )}
      </div>

      {/* ── Main Content ── */}
      <div style={{ width: "100%", padding: "40px 40px 60px", boxSizing: "border-box" }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: result ? "1fr 1fr" : "1fr",
          gap: 32,
          width: "100%",
        }}>

          {/* ── LEFT: Upload + Status ── */}
          <div style={{ width: "100%" }}>
            <h2 style={{ fontSize: 12, fontWeight: 600, color: "#64748b", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 16 }}>
              Input Video
            </h2>

            {/* Drop zone */}
            {!file ? (
              <div
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: `2px dashed ${dragOver ? "#38bdf8" : "#1e293b"}`,
                  borderRadius: 16, padding: "80px 40px",
                  textAlign: "center", cursor: "pointer",
                  background: dragOver ? "#0f172a" : "transparent",
                  transition: "all 0.2s", width: "100%", boxSizing: "border-box",
                }}
              >
                <div style={{ fontSize: 56, marginBottom: 16 }}>📹</div>
                <p style={{ color: "#94a3b8", margin: 0, fontSize: 16 }}>
                  Drop a blurry or low-quality video here
                </p>
                <p style={{ color: "#475569", margin: "8px 0 20px", fontSize: 13 }}>
                  MP4 · AVI · MOV · MKV · WebM supported
                </p>
                <span style={{
                  background: "linear-gradient(135deg, #38bdf8, #818cf8)",
                  borderRadius: 8, color: "#fff", padding: "10px 24px",
                  fontWeight: 600, fontSize: 14, display: "inline-block",
                }}>Browse File</span>
                <input ref={fileInputRef} type="file" accept="video/*"
                  style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
              </div>
            ) : (
              <div style={{ width: "100%" }}>
                <video src={videoUrl} controls style={{
                  width: "100%", borderRadius: 12,
                  border: "1px solid #1e293b", background: "#000", display: "block",
                }} />
                <div style={{
                  marginTop: 12, padding: "14px 18px", background: "#0f172a",
                  borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "space-between",
                  gap: 12, flexWrap: "wrap",
                }}>
                  <div>
                    <p style={{ margin: 0, fontSize: 13, color: "#94a3b8" }}>{file.name}</p>
                    <p style={{ margin: 0, fontSize: 11, color: "#475569" }}>
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  {status === "idle" && (
                    <button onClick={handleUpload} style={{
                      background: "linear-gradient(135deg, #38bdf8, #818cf8)",
                      border: "none", borderRadius: 8, color: "#fff",
                      cursor: "pointer", padding: "11px 26px",
                      fontWeight: 600, fontSize: 14, whiteSpace: "nowrap",
                    }}>
                      Enhance & Analyze →
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Status card */}
            {status !== "idle" && (
              <div style={{
                marginTop: 20, background: "#0f172a", borderRadius: 16, padding: 24,
                border: `1px solid ${status === "done" ? "#1d4ed8" : status === "error" ? "#7f1d1d" : "#1e293b"}`,
                width: "100%", boxSizing: "border-box",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                  {(status === "processing" || status === "uploading") ? (
                    <ProgressRing progress={fakeProgress} />
                  ) : (
                    <div style={{
                      width: 80, height: 80, borderRadius: "50%", flexShrink: 0,
                      background: status === "done" ? "#0c4a6e" : "#450a0a",
                      display: "flex", alignItems: "center", justifyContent: "center", fontSize: 34,
                    }}>
                      {status === "done" ? "✅" : "❌"}
                    </div>
                  )}
                  <div>
                    <p style={{ margin: 0, fontWeight: 600, fontSize: 17 }}>{statusLabel}</p>
                    {(status === "processing" || status === "uploading") && (
                      <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: 13 }}>
                        SRGAN enhancing frames · InceptionV3 classifying actions
                      </p>
                    )}
                    {status === "error" && (
                      <p style={{ margin: "4px 0 0", color: "#f87171", fontSize: 13 }}>{error}</p>
                    )}
                  </div>
                </div>

                {(status === "processing" || status === "uploading") && (
                  <div style={{ marginTop: 20 }}>
                    <div style={{ background: "#1e293b", borderRadius: 99, height: 5 }}>
                      <div style={{
                        width: `${fakeProgress}%`, height: "100%",
                        background: "linear-gradient(90deg, #38bdf8, #818cf8)",
                        borderRadius: 99, transition: "width 0.3s ease",
                      }} />
                    </div>
                    <p style={{ margin: "6px 0 0", fontSize: 12, color: "#38bdf8", textAlign: "right" }}>
                      {Math.round(fakeProgress)}%
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Pipeline info */}
            <div style={{
              marginTop: 20, background: "#0f172a", borderRadius: 12,
              padding: "18px 22px", border: "1px solid #1e293b",
              width: "100%", boxSizing: "border-box",
            }}>
              <p style={{ margin: "0 0 12px", fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1 }}>
                Pipeline
              </p>
              {[
                ["🔍", "SRGAN", "4× super-resolution on each frame"],
                ["🧠", "InceptionV3", "Action classification across 101 classes"],
                ["📦", "Output", "Full-resolution MP4 + action labels"],
              ].map(([icon, label, desc]) => (
                <div key={label} style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                  <span style={{ fontSize: 16 }}>{icon}</span>
                  <p style={{ margin: 0, fontSize: 13 }}>
                    <strong style={{ color: "#cbd5e1" }}>{label} </strong>
                    <span style={{ color: "#64748b" }}>{desc}</span>
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* ── RIGHT: Results ── */}
          {result && (
            <div style={{ width: "100%" }}>
              <h2 style={{ fontSize: 12, fontWeight: 600, color: "#64748b", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 16 }}>
                Results
              </h2>

              {/* Top action */}
              <div style={{
                background: "linear-gradient(135deg, #0c4a6e22, #1e1b4b22)",
                border: "1px solid #1d4ed8", borderRadius: 16,
                padding: "22px 26px", marginBottom: 16,
              }}>
                <p style={{ margin: "0 0 4px", fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>
                  Detected Action
                </p>
                <p style={{ margin: "0 0 10px", fontSize: 32, fontWeight: 700, color: "#38bdf8" }}>
                  {result.top_action}
                </p>
                <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 13, color: "#94a3b8" }}>
                    Confidence: <strong style={{ color: "#f1f5f9" }}>{(result.confidence * 100).toFixed(1)}%</strong>
                  </span>
                  <span style={{ fontSize: 13, color: "#94a3b8" }}>
                    Resolution: <strong style={{ color: "#f1f5f9" }}>{result.original_resolution} → {result.enhanced_resolution}</strong>
                  </span>
                  <span style={{ fontSize: 13, color: "#94a3b8" }}>
                    Frames: <strong style={{ color: "#f1f5f9" }}>{result.total_frames}</strong>
                  </span>
                </div>
              </div>

              {/* Top 5 */}
              <div style={{
                background: "#0f172a", borderRadius: 16,
                padding: "22px 26px", marginBottom: 16,
              }}>
                <p style={{ margin: "0 0 16px", fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>
                  Top 5 Predictions
                </p>
                {result.top5.map((pred, i) => (
                  <ConfidenceBar key={pred.action} label={pred.action} value={pred.confidence} rank={i} />
                ))}
              </div>

              {/* Download */}
              <button onClick={handleDownload} style={{
                width: "100%", padding: "15px",
                background: "linear-gradient(135deg, #0284c7, #4f46e5)",
                border: "none", borderRadius: 12, color: "#fff",
                fontWeight: 700, fontSize: 16, cursor: "pointer",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
              }}>
                ⬇️ Download Enhanced Video ({result.enhanced_resolution})
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}