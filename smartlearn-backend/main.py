import os
import re

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services import rag

app = FastAPI(title="SmartLearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Day 3 documents store: one rich record per chat_id
documents: dict[str, dict] = {}


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(chat_id: str = Query(...), file: UploadFile = File(...)):
    """Upload a PDF and build a RAG-ready document record.

    The visible response shape stays Day-2-compatible:
        {"status": "ok", "filename": "...", "pages": N, "characters": N}
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Extract pages first so we can validate text is readable
    pages = rag.extract_pages_from_bytes_for_rag(pdf_bytes)
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in PDF. OCR is not supported.",
        )

    # Build the full Day 3 RAG record (chunks + FAISS index + empty history)
    try:
        document = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            pages=pages,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to prepare RAG index: {e}",
        )

    documents[chat_id] = document
    return rag.build_upload_response(document)


@app.get("/documents/{chat_id}/file")
def serve_pdf(chat_id: str):
    """Serve the uploaded PDF for the current chat session.

    Returns 404 if *chat_id* is unknown or the saved file is missing.
    """
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{chat_id}'. Upload a PDF first.",
        )

    file_path = document.get("saved_pdf_path")
    if file_path is None or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found for chat_id '{chat_id}'.",
        )

    return FileResponse(file_path, media_type="application/pdf")


class ChatRequest(BaseModel):
    chat_id: str = Field(default="day2-demo")
    message: str = Field(min_length=2, max_length=2000)


@app.post("/chat")
def chat(body: ChatRequest):
    """Answer a question with RAG retrieval and multi-turn history.

    Keeps the Day 2 request shape:
        {"chat_id": "...", "message": "..."}

    Returns:
        {"answer": "...", "citations": [4, 8], "sources": [...]}
    """
    document = documents.get(body.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No document found for chat_id '{body.chat_id}'. "
                "Upload a PDF first."
            ),
        )

    try:
        result = rag.answer_chat_turn(
            document,
            body.message,
            top_k=3,
            candidate_pool=60,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Upstream AI service failed")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
