import React, { useState } from "react";
import { uploadPDF, askQuestion } from "./api.js";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function handleUpload(fileToUpload) {
    try {
      setStatus("uploading");
      setError("");
      setAnswer(null);
      setUpload(await uploadPDF(fileToUpload));
    } catch (e) {
      setError(e.message || "Upload failed");
    } finally {
      setStatus("idle");
    }
  }

  async function handleAsk(text) {
    try {
      setStatus("asking");
      setError("");
      setAnswer(await askQuestion(text));
    } catch (e) {
      setError(e.message || "Chat failed");
    } finally {
      setStatus("idle");
    }
  }

  const isBusy = status !== "idle";

  return (
    <div className="app-container">

      {/* ====== Header ====== */}
      <header className="app-header">
        <h1>📚 SmartLearn Lite</h1>
        <p className="subtitle">Upload a PDF and ask questions about its content</p>
      </header>

      {/* ====== Upload Card ====== */}
      <div className="card">
        <div className="card-title">
          <span className="step">1</span> Upload a PDF
        </div>

        <label className={`file-upload-area ${file ? "has-file" : ""}`}>
          <div className="file-icon">{file ? "📄" : "📁"}</div>
          {file ? (
            <span className="file-name">{file.name}</span>
          ) : (
            <>
              <span className="file-label">Click to choose a file</span>
              <span className="file-hint">or drag and drop · PDF only · Max 30 pages</span>
            </>
          )}
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </label>

        <button
          className="btn btn-primary"
          disabled={!file || isBusy}
          onClick={() => handleUpload(file)}
        >
          {status === "uploading" ? (
            <><span className="spinner" /> Uploading…</>
          ) : (
            "Upload PDF"
          )}
        </button>

        {upload && (
          <div className="upload-info">
            <span className="tag">📄 {upload.filename}</span>
            <span className="tag">📃 {upload.pages} pages</span>
            <span className="tag">🔤 {upload.characters} chars</span>
          </div>
        )}
      </div>

      {/* ====== Chat Card ====== */}
      {upload && (
        <div className="card">
          <div className="card-title">
            <span className="step">2</span> Ask a Question
          </div>

          <textarea
            placeholder='e.g. "What is the main conclusion of this document?"'
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (message.trim() && !isBusy) handleAsk(message.trim());
              }
            }}
          />

          <button
            className="btn btn-primary"
            disabled={!message.trim() || isBusy}
            onClick={() => handleAsk(message.trim())}
          >
            {status === "asking" ? (
              <><span className="spinner" /> Thinking…</>
            ) : (
              "Ask Question"
            )}
          </button>
        </div>
      )}

      {/* ====== Error ====== */}
      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* ====== Answer ====== */}
      {answer && (
        <div className="answer-card">
          <div className="answer-label">📝 Answer</div>
          <div className="answer-text">{answer.answer}</div>

          {answer.citations.length > 0 && (
            <div className="citations-row">
              <span className="citations-label">Sources</span>
              {answer.citations.map((page) => (
                <span className="page-chip" key={page}>
                  📌 Page {page}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

    </div>
  );
}

export default App;
