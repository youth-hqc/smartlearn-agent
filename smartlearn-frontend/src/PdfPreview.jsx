import React, { useState, useEffect } from "react";
import { API } from "./api.js";

function getDocumentFileURL(chatId, page = 1) {
  return `${API}/documents/${encodeURIComponent(chatId)}/file#page=${page}`;
}

/**
 * PdfPreview — iframe-based PDF viewer that follows page citations.
 *
 * Before rendering the iframe we check whether the backend still has the
 * uploaded file.  If the backend was restarted the in-memory document is
 * gone, so we show a "re-upload" hint instead of the raw 404 JSON that
 * would otherwise appear inside the iframe.
 */
function PdfPreview({ upload, activePage, previewKey }) {
  const [fileReady, setFileReady] = useState(null); // null=checking, true=ok, false=gone

  useEffect(() => {
    if (!upload) {
      setFileReady(null);
      return;
    }
    let cancelled = false;
    const chatId = upload.chat_id || "day2-demo";
    // Quick check to see if the PDF is still on the server.
    // Use GET with a tiny Range so we don't download the whole file.
    fetch(`${API}/documents/${encodeURIComponent(chatId)}/file`, {
      method: "GET",
      headers: { "Range": "bytes=0-0" },
    })
      .then((res) => { if (!cancelled) setFileReady(res.ok || res.status === 206); })
      .catch(() => { if (!cancelled) setFileReady(null); }); // null = assume ok, let iframe try
    return () => { cancelled = true; };
  }, [upload, previewKey]);

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

      {fileReady === false ? (
        <div className="preview-placeholder">
          <div className="preview-placeholder-icon">🔄</div>
          <p>PDF no longer available on the server</p>
          <p className="preview-hint">
            The backend may have restarted. Please re-upload the PDF.
          </p>
        </div>
      ) : (
        <iframe
          key={previewKey}
          src={url}
          className="preview-iframe"
          title="PDF Preview"
        />
      )}
    </div>
  );
}

export default PdfPreview;
