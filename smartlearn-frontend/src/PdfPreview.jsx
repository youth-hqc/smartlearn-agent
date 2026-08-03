import React from "react";
import { API } from "./api.js";

/**
 * Build the backend PDF file URL for the current chat session.
 * Appends #page=N for in-PDF page jumps.
 */
function getDocumentFileURL(chatId, page = 1) {
  return `${API}/documents/${encodeURIComponent(chatId)}/file#page=${page}`;
}

/**
 * PdfPreview — iframe-based PDF viewer that follows page citations.
 *
 * Props:
 *   upload      — the upload response object, or null
 *   activePage  — the page number to show
 *   previewKey  — bumps when a new file is uploaded (forces iframe remount)
 */
function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <div className="preview-panel">
        <div className="preview-placeholder">
          <div className="preview-placeholder-icon">📄</div>
          <p>Upload a PDF to preview</p>
          <p className="preview-hint">
            Ask a question, then click a page citation to jump here
          </p>
        </div>
      </div>
    );
  }

  const chatId = upload.chat_id || "day2-demo";
  const url = getDocumentFileURL(chatId, activePage);

  return (
    <div className="preview-panel">
      <div className="preview-header">
        <span className="preview-label">
          📄 {upload.filename} — Page {activePage}
        </span>
      </div>
      <iframe
        key={previewKey}
        src={url}
        className="preview-iframe"
        title="PDF Preview"
      />
    </div>
  );
}

export default PdfPreview;
