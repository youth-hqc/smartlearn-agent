// In production (Railway) frontend + backend share the same domain,
// so an empty string makes all requests relative.  For local dev set
// VITE_API_URL=http://localhost:8000 in smartlearn-frontend/.env
export const API = import.meta.env.VITE_API_URL || "";
export const CHAT_ID = "day2-demo";

async function readJSON(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(
    `${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`,
    { method: "POST", body: formData },
  );
  const data = await readJSON(response);
  // Attach chat_id so the preview component can build the file URL
  return { ...data, chat_id: CHAT_ID };
}

export async function askQuestion(message) {
  const response = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, chat_id: CHAT_ID }),
  });
  return readJSON(response);
}
