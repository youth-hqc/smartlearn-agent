import React, { useState } from "react";
import { askQuestion } from "./api.js";

/**
 * ChatPanel — multi-turn message list with citation buttons.
 *
 * Props:
 *   enabled       — whether the panel is visible (upload succeeded)
 *   onBusy        — callback(busy) so parent can disable upload during ask
 *   disabled      — external disable signal (e.g. another operation in flight)
 *   onJumpToPage  — callback(page) when user clicks a citation
 */
function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  // Multi-turn message list: { role: "user"|"assistant", content, citations?, sources? }
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  if (!enabled) {
    return null;
  }

  const isBusy = loading || disabled;

  async function handleAsk(text) {
    if (!text.trim() || isBusy) return;
    setLoading(true);
    setError("");
    if (onBusy) onBusy(true);

    // Append user message immediately
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setMessage("");

    try {
      const result = await askQuestion(text);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          citations: result.citations || [],
          sources: result.sources || [],
        },
      ]);
    } catch (e) {
      setError(e.message || "Chat failed");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, something went wrong.",
          citations: [],
          sources: [],
        },
      ]);
    } finally {
      setLoading(false);
      if (onBusy) onBusy(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Ask a question about the uploaded document.</p>
            <p className="chat-empty-hint">
              Citations will appear as clickable page numbers.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg ${msg.role}`}>
            <div className="chat-msg-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="chat-msg-text">{msg.content}</div>

            {msg.role === "assistant" && msg.citations.length > 0 && (
              <div className="chat-citations">
                <span className="chat-citations-label">Pages</span>
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    className="chat-page-chip"
                    onClick={() => onJumpToPage && onJumpToPage(page)}
                    title={`Jump to page ${page}`}
                  >
                    📌 {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-msg assistant">
            <div className="chat-msg-role">Assistant</div>
            <div className="chat-msg-text">
              <span className="spinner-dark" /> Thinking…
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="chat-error" role="alert">
          {error}
        </div>
      )}

      <div className="chat-input-row">
        <textarea
          placeholder='e.g. "What is the main conclusion?"'
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleAsk(message.trim());
            }
          }}
          rows={2}
        />
        <button
          className="btn btn-primary chat-send-btn"
          disabled={!message.trim() || isBusy}
          onClick={() => handleAsk(message.trim())}
        >
          {loading ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}

export default ChatPanel;
