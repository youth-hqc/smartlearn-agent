import React, { useState } from "react";
import { uploadPDF } from "./api.js";
import PdfPreview from "./PdfPreview.jsx";
import ChatPanel from "./ChatPanel.jsx";

function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [previewKey, setPreviewKey] = useState(0);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function handleUpload(fileToUpload) {
    try {
      setStatus("uploading");
      setError("");
      setUpload(null);
      setUpload(await uploadPDF(fileToUpload));
      setActivePage(1);                // reset to page 1
      setPreviewKey((k) => k + 1);     // remount preview + chat
    } catch (e) {
      setError(e.message || "Upload failed");
    } finally {
      setStatus("idle");
    }
  }

  function handleJumpToPage(page) {
    setActivePage(page);
  }

  function handleChatBusy(busy) {
    setStatus(busy ? "asking" : "idle");
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
              <span className="file-hint">or drag and drop · PDF only</span>
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

      {/* ====== Error ====== */}
      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* ====== Two-column workspace (preview + chat) ====== */}
      {upload && (
        <div className="workspace">
          <PdfPreview
            upload={upload}
            activePage={activePage}
            previewKey={previewKey}
          />
          <ChatPanel
            key={previewKey}  // remount on new upload to clear old messages
            enabled={!!upload}
            onBusy={handleChatBusy}
            disabled={isBusy}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      )}

    </div>
  );
}

export default App;
