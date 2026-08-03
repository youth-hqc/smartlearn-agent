"""
RAG pipeline helpers: text cleaning, page loading, chunking, embeddings, and artifact saving.

Sections:
    1. Text cleaning & page loading
    2. Chunking (paragraph, character, character_overlap)
    3. Embedding pipeline & artifact management
    4. LangChain recursive splitter (Appendix A)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Text cleaning & page loading
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalize one page of extracted PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy
    line-break characters so downstream chunking sees predictable input.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove soft hyphens (U+00AD)
    text = text.replace("­", "")

    # Normalize common PDF noise: vertical tabs, form feeds, carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\v", "\n").replace("\f", "\n")

    # Collapse repeated whitespace (but keep single newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ newlines into max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace on each line, then the whole block
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def extract_pages_for_rag(pdf_path) -> list[dict]:
    """Read a PDF page by page and return readable ``[{page, text}]`` records.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    list[dict]
        Each dict has keys ``page`` (int, 1-based original PDF page number)
        and ``text`` (str, cleaned extracted text).  Empty pages are omitted.
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})

    return records


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def save_json(obj, path):
    """Save a Python object to a UTF-8 JSON file, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def load_json(path):
    """Read a saved JSON artifact back into Python."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------


def preview_records(records: list[dict], columns: list[str], rows: int = 5):
    """Show a small notebook table for chosen columns.

    Requires ``pandas`` to be installed in the notebook kernel.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for preview_records") from exc

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable = [c for c in columns if c in frame.columns]
    return frame[usable].head(rows)


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------


def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into smaller pieces.

    Prefers natural boundaries (newlines, then spaces) and avoids splitting
    in the middle of a word whenever possible.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        # Try to cut at a natural boundary within the last 20 % of the window
        window = remaining[:chunk_size]
        cut = len(window)

        for sep in ("\n\n", "\n", ". ", " "):
            pos = window.rfind(sep)
            if pos > chunk_size * 0.5:
                cut = pos + len(sep)
                break
        else:
            # No good boundary — fall back to a hard character cut
            cut = chunk_size
            # Back up to the last space if one exists
            last_space = window.rfind(" ")
            if last_space > chunk_size * 0.5:
                cut = last_space + 1

        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def chunk_by_paragraph(
    records: list[dict],
    chunk_size: int = 700,
    overlap: int = 0,  # kept for uniform signature; not used in paragraph mode
) -> list[dict]:
    """Convert page records into paragraph-level chunks.

    Preserves paragraph boundaries as much as possible.  When a single
    paragraph exceeds *chunk_size* it is split into smaller pieces via
    :func:`slice_long_text`.
    """
    chunks: list[dict] = []
    chunk_id = 0

    for record in records:
        page = record["page"]
        # Split the page text into paragraphs (double-newline separated)
        paragraphs = re.split(r"\n\s*\n", record["text"].strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        for para in paragraphs:
            if len(para) <= chunk_size:
                chunk_id += 1
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": para,
                        "chunk_mode": "paragraph",
                    }
                )
            else:
                # Oversized paragraph — split further
                sub_pieces = slice_long_text(para, chunk_size)
                for piece in sub_pieces:
                    chunk_id += 1
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "page": page,
                            "text": piece,
                            "chunk_mode": "paragraph",
                        }
                    )

    return chunks


def chunk_by_characters(
    records: list[dict],
    chunk_size: int = 700,
    overlap: int = 0,
) -> list[dict]:
    """Create plain fixed-size sliding-window chunks.

    When *overlap* is 0 this is a simple non-overlapping character split.
    When *overlap* > 0 each window shares *overlap* characters with the
    previous window.
    """
    chunks: list[dict] = []
    chunk_id = 0

    # Flatten all pages into a single text stream while tracking page boundaries
    for record in records:
        page = record["page"]
        text = record["text"]
        step = max(1, chunk_size - overlap)
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk_id += 1
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": piece,
                        "chunk_mode": (
                            "character_overlap" if overlap > 0 else "character"
                        ),
                    }
                )
            start += step

    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Select a chunking strategy and return a uniform chunk schema.

    Parameters
    ----------
    records : list[dict]
        Page records from :func:`extract_pages_for_rag`.
    chunk_mode : str
        One of ``"paragraph"``, ``"character"``, ``"character_overlap"``,
        or ``"langchain_recursive"`` (Appendix A).
    chunk_size : int
        Maximum characters per chunk.
    overlap : int
        Character overlap between adjacent windows (only meaningful for
        ``character_overlap`` and ``langchain_recursive``).

    Returns
    -------
    list[dict]
        Each chunk has ``chunk_id``, ``page``, ``text``, and ``chunk_mode``.
    """
    _mode = chunk_mode.lower()

    if _mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size=chunk_size, overlap=overlap)

    if _mode in ("character", "character_overlap"):
        return chunk_by_characters(records, chunk_size=chunk_size, overlap=overlap)

    if _mode == "langchain_recursive":
        return chunk_with_langchain_recursive(
            records, chunk_size=chunk_size, chunk_overlap=overlap
        )

    raise ValueError(
        f"Unknown chunk_mode '{chunk_mode}'. "
        f"Supported: paragraph, character, character_overlap, langchain_recursive"
    )


# ---------------------------------------------------------------------------
# 3. Embedding pipeline
# ---------------------------------------------------------------------------

# Module-level cache so load_model reuses the same instance
_model_cache: dict = {}


def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix.

    >>> model_tag("sentence-transformers/all-MiniLM-L6-v2")
    'all_MiniLM_L6_v2'
    """
    # Take the last segment after the final /
    short = model_name.rsplit("/", 1)[-1]
    # Replace hyphens and dots with underscores; collapse runs
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", short)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


def _find_local_model(model_name: str,) -> Optional[Path]:
    """Search common local model directories for *model_name*."""
    candidates: list[Path] = []

    # 1. Notebook artifact location
    candidates.append(
        Path("Day3/artifacts/hf_models") / model_name.rsplit("/", 1)[-1]
    )

    # 2. Backend artifact location
    candidates.append(
        Path("smartlearn-backend/artifacts/rag/hf_models")
        / model_name.rsplit("/", 1)[-1]
    )

    # 3. HuggingFace cache (default location)
    hf_home = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
    candidates.append(
        Path(hf_home) / ("models--" + model_name.replace("/", "--"))
    )

    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            # Basic sanity: look for config files
            if (
                (candidate / "config_sentence_transformers.json").exists()
                or (candidate / "config.json").exists()
            ):
                return candidate
            # Also check for snapshots/ subdir (HF cache layout)
            snapshots = candidate / "snapshots"
            if snapshots.exists():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if snap.is_dir():
                        return snap

    return None


def resolve_model_source(
    model_name: str,
    artifact_root: Optional[Path] = None,
) -> str:
    """Prefer a local cached model folder; fall back to the canonical name.

    Returns a path string if a local copy is found, otherwise the original
    *model_name* (which ``SentenceTransformer`` will download from HuggingFace).
    """
    local = _find_local_model(model_name)
    if local is not None:
        return str(local)
    return model_name


def get_device() -> str:
    """Return ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(
    model_name: str,
    device: Optional[str] = None,
):
    """Create or reuse one ``SentenceTransformer`` model instance.

    The model is cached at module level so repeated calls inside the same
    Python process return the same object.
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = get_device()

    cache_key = (model_name, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    source = resolve_model_source(model_name)
    model = SentenceTransformer(
        source,
        device=device,
    )
    _model_cache[cache_key] = model
    return model


def embed_texts(
    texts: list[str],
    model=None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> "np.ndarray":  # noqa: F821
    """Encode a list of texts into normalized ``float32`` vectors.

    Parameters
    ----------
    texts : list[str]
        The chunk texts to embed.
    model :
        An existing ``SentenceTransformer`` instance.  If ``None`` a model
        is loaded via :func:`load_model`.
    model_name : str
        Model identifier, used only when *model* is ``None``.
    batch_size : int
        Batch size passed to ``model.encode()``.
    device : str or None
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).

    Returns
    -------
    numpy.ndarray
        Shape ``(len(texts), dim)``, dtype ``float32``, L2-normalized.
    """
    import numpy as np

    if model is None:
        model = load_model(model_name, device=device)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.astype(np.float32)


def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    artifact_root,
) -> dict:
    """Decide where pages, chunks, embeddings, manifests, and indexes should be saved.

    Returns a dict with keys:
        raw_pages, chunks, embeddings, manifest, index
    """
    root = Path(artifact_root)
    tag = model_tag(model_name)

    raw_dir = root / "raw_pages"
    chunk_dir = root / "chunks"
    embed_dir = root / "embeddings"

    return {
        "raw_pages": raw_dir / f"{document_id}_pages.json",
        "chunks": chunk_dir / f"{document_id}_{chunk_mode}.json",
        "embeddings": embed_dir / f"{document_id}_{chunk_mode}_{tag}.npy",
        "manifest": embed_dir / f"{document_id}_{chunk_mode}_{tag}.manifest.json",
        "index": embed_dir / f"{document_id}_{chunk_mode}_{tag}.faiss",
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root=None,
    force_rebuild: bool = False,
) -> dict:
    """Build (or reuse) the full pages → chunks → embeddings → manifest bundle.

    Parameters
    ----------
    document_id : str
        Short id for this document (e.g. ``"pdf1"``).
    pdf_name : str
        Original filename (recorded in the manifest only).
    pages : list[dict]
        Output of :func:`extract_pages_for_rag`.
    chunk_mode : str
        Chunking strategy.
    model_name : str
        HuggingFace model id.
    chunk_size : int
    overlap : int
    batch_size : int
    artifact_root : str or Path
        Where to write the artifact tree (e.g. ``Day3/artifacts``).
    force_rebuild : bool
        If ``True``, rebuild even when cached artifacts exist.

    Returns
    -------
    dict
        ``{"manifest": {...}, "chunks": [...], "embeddings": np.ndarray}``
    """
    import numpy as np

    root = Path(artifact_root) if artifact_root else Path("artifacts")
    paths = artifact_paths_for(document_id, chunk_mode, model_name, root)

    # --- Raw pages ---
    save_json(pages, paths["raw_pages"])

    # --- Chunks ---
    chunks = build_chunks(pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap)
    save_json(chunks, paths["chunks"])

    # --- Embeddings ---
    # Reuse cached embeddings when the chunk count still matches
    cached_manifest = None
    if paths["manifest"].exists():
        cached_manifest = load_json(paths["manifest"])

    if (
        not force_rebuild
        and paths["embeddings"].exists()
        and cached_manifest is not None
        and cached_manifest.get("num_chunks") == len(chunks)
        and cached_manifest.get("chunk_mode") == chunk_mode
        and cached_manifest.get("model_name") == model_name
    ):
        embeddings = np.load(str(paths["embeddings"]))
        device_used = cached_manifest.get("device", "cpu")
    else:
        device_used = get_device()
        model = load_model(model_name, device=device_used)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts, model=model, batch_size=batch_size)

        paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
        np.save(str(paths["embeddings"]), embeddings)

    # --- Manifest ---
    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": device_used,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, paths["manifest"])

    return {
        "manifest": manifest,
        "chunks": chunks,
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# Appendix A — LangChain RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------


def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    separators=None,
) -> list[dict]:
    """Split each page with LangChain's ``RecursiveCharacterTextSplitter``.

    Uses a separator priority of double-newline → single newline → space →
    character-level fallback, which often produces cleaner chunk boundaries
    on noisy PDF text.

    Requires ``langchain-text-splitters`` to be installed.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "langchain-text-splitters is required for chunk_with_langchain_recursive. "
            "Install it with: pip install langchain-text-splitters"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )

    chunks: list[dict] = []
    chunk_id = 0

    for record in pages:
        page = record["page"]
        if not record["text"].strip():
            continue

        docs = splitter.create_documents([record["text"]])
        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            chunk_id += 1
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page,
                    "text": text,
                    "chunk_mode": "langchain_recursive",
                }
            )

    return chunks
