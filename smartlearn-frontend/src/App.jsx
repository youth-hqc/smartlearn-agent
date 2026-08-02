import { useState } from "react";
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
    <main>
      <h1>SmartLearn Lite</h1>

      {/* ---------- Upload ---------- */}
      <section>
        <label htmlFor="pdf-file">Choose PDF</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button
          disabled={!file || isBusy}
          onClick={() => handleUpload(file)}
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>

        {upload && (
          <p>
            Uploaded: {upload.filename} ({upload.pages} pages, {upload.characters} chars)
          </p>
        )}
      </section>

      {/* ---------- Chat ---------- */}
      {upload && (
        <section>
          <label htmlFor="message">Message</label>
          <textarea
            id="message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />

          <button
            disabled={!message.trim() || isBusy}
            onClick={() => handleAsk(message.trim())}
          >
            {status === "asking" ? "Asking…" : "Ask"}
          </button>
        </section>
      )}

      {/* ---------- Error ---------- */}
      {error && <p role="alert">{error}</p>}

      {/* ---------- Answer ---------- */}
      {answer && (
        <section>
          <p>{answer.answer}</p>
          <div>
            {answer.citations.map((page) => (
              <span key={page}>Page {page}</span>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

export default App;
