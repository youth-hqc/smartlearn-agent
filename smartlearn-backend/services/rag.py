"""
RAG pipeline helpers: text cleaning, page loading, chunking, embeddings,
FAISS retrieval, local answer extraction, evaluation, and artifact saving.

Sections:
    1. Text cleaning & page loading
    2. Chunking (paragraph, character, character_overlap)
    3. Embedding pipeline & artifact management
    4. FAISS index helpers
    5. Retrieval & local answer extraction
    6. Project-facing wrappers
    7. Evaluation helpers
    A. LangChain recursive splitter
    B. Chroma collection (optional)
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


# ---------------------------------------------------------------------------
# 4. FAISS index helpers
# ---------------------------------------------------------------------------


def relative_path_str(path, base=None):
    """Return a shorter display path relative to *base* when possible."""
    path = Path(path)
    if base is not None:
        base = Path(base)
        try:
            return str(path.relative_to(base))
        except ValueError:
            pass
    return str(path)


def build_faiss_index(
    embeddings: "np.ndarray",  # noqa: F821
    index_type: str = "flat",
    nlist: int | None = None,
) -> "faiss.Index":  # noqa: F821
    """Create a FAISS index from *embeddings*.

    Parameters
    ----------
    index_type : str
        ``"flat"`` — brute-force exact search (default, best for <10K chunks).
        ``"ivf"`` — inverted-file approximate search (faster for 10K+ chunks).
    nlist : int or None
        Number of IVF clusters.  Auto-computed as ``4 * sqrt(N)`` when None.

    Because embeddings are L2-normalized, inner-product equals cosine similarity.
    """
    import faiss
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = int(embeddings.shape[1])
    n = embeddings.shape[0]

    if index_type == "ivf" and n >= 100:
        if nlist is None:
            nlist = max(4, int(4 * np.sqrt(n)))
        nlist = min(nlist, n // 2)  # can't have more clusters than data/2

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = max(1, nlist // 10)  # search ~10% of clusters
        return index

    # Fall back to flat (exact) for small collections
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index, index_path):
    """Write a FAISS index to a binary ``.faiss`` file."""
    import faiss

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path) -> "faiss.Index":  # noqa: F821
    """Read a saved ``.faiss`` binary file back into memory."""
    import faiss

    return faiss.read_index(str(index_path))


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path=None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Build (or reuse) the chunks, embeddings, FAISS index, and manifest.

    Returns a bundle with keys:
        ``manifest``, ``chunks``, ``embeddings``, ``index``, ``paths``
    """
    import numpy as np

    root = Path(artifact_root) if artifact_root else Path("artifacts")
    paths = artifact_paths_for(document_id, chunk_mode, model_name, root)

    # --- Ensure chunks + embeddings exist (reuse when signature matches) ---
    artifact_bundle = ensure_artifacts(
        document_id=document_id,
        pdf_name=pdf_name,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )
    chunks = artifact_bundle["chunks"]
    embeddings = artifact_bundle["embeddings"]

    # --- Build or load FAISS index ---
    if paths["index"].exists():
        index = load_faiss_index(paths["index"])
        if index.ntotal != len(chunks):
            # Chunk count changed -- rebuild
            index = build_faiss_index(embeddings)
            save_faiss_index(index, paths["index"])
    else:
        index = build_faiss_index(embeddings)
        save_faiss_index(index, paths["index"])

    paths["index"].parent.mkdir(parents=True, exist_ok=True)

    return {
        "manifest": artifact_bundle["manifest"],
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# 5. Retrieval & local answer extraction
# ---------------------------------------------------------------------------


def keyword_set(text: str) -> set[str]:
    """Return a lightweight set of lexical tokens for simple reranking."""
    import re

    if not text:
        return set()
    # Lowercase, keep alphanumeric tokens of length >= 2
    tokens = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
    return set(tokens)


# ---------------------------------------------------------------------------
# BM25 sparse retriever (pure Python, no extra deps)
# ---------------------------------------------------------------------------


class BM25SparseRetriever:
    """Pure-Python BM25 for keyword-precise retrieval.

    BM25 excels at matching exact terms (like "Post-training" or "SmolLM3")
    that dense embeddings sometimes miss.  No model inference — purely
    CPU-based term-frequency math.
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        import math
        import re
        from collections import Counter

        self.k1 = k1
        self.b = b
        self.chunks = chunks

        # Tokenize each chunk
        self._docs: list[list[str]] = []
        for c in chunks:
            tokens = re.findall(r"[a-zA-Z0-9]{2,}", c["text"].lower())
            self._docs.append(tokens)

        # Average document length
        self._avgdl = sum(len(d) for d in self._docs) / max(1, len(self._docs))

        # Document frequency (df) for IDF
        df: dict[str, int] = Counter()
        for doc in self._docs:
            for token in set(doc):
                df[token] += 1

        N = len(self._docs)
        self._idf: dict[str, float] = {}
        for token, freq in df.items():
            self._idf[token] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))

        # Term frequency per document
        self._tf: list[Counter] = [Counter(doc) for doc in self._docs]
        self._doc_len = [len(doc) for doc in self._docs]

    def search(self, query: str, top_k: int = 60) -> list[tuple[int, float]]:
        """Return top-k ``(chunk_index, bm25_score)`` pairs."""
        import re

        q_tokens = re.findall(r"[a-zA-Z0-9]{2,}", query.lower())
        if not q_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for i, doc_tf in enumerate(self._tf):
            score = 0.0
            for token in q_tokens:
                idf = self._idf.get(token, 0)
                if idf == 0:
                    continue
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (self._doc_len[i] / self._avgdl)
                )
                score += idf * numerator / denominator
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Query expansion (keyword-based, no LLM overhead)
# ---------------------------------------------------------------------------


def expand_query_variants(question: str) -> list[str]:
    """Return a few keyword-focused query variants for better recall.

    Generates short-form and keyword-only variants so a single question can
    match chunks that use different phrasing.  No model calls — pure regex.
    """
    import re

    variants = [question]

    # Variant 1: strip common question prefixes
    short = re.sub(
        r"^(what is|which|who|where|when|why|how|name the|list the|describe|explain)\s+",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    )
    if short != question and len(short) > 8:
        variants.append(short)

    # Variant 2: extract capitalized terms and 4+ char words as keywords
    caps = re.findall(r"[A-Z][a-zA-Z0-9+-]{2,}", question)
    longs = re.findall(r"[a-zA-Z]{4,}", question.lower())
    keywords = list(dict.fromkeys(caps + longs))  # dedup, keep order
    if len(keywords) >= 2:
        variants.append(" ".join(keywords[:8]))

    return list(dict.fromkeys(variants))  # dedup, keep order


def _ensure_query_model(
    model_name: str,
    device: str | None = None,
):
    """Load (or reuse) the embedding model for query encoding."""
    return load_model(model_name, device=device)


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
    hybrid: bool = True,
    bm25_weight: float = 0.15,
    use_reranker: bool = False,
) -> list[dict]:
    """Search an in-memory index bundle and return top-k hits.

    Parameters
    ----------
    hybrid : bool
        When True (default), fuse BM25 keyword scores with FAISS dense scores.
        The BM25 component catches exact-term matches that dense embeddings
        may miss (e.g. "Post-training", "SmolLM3").
    bm25_weight : float
        Weight of BM25 in the hybrid score (0.0 = pure dense, 1.0 = pure BM25).
    use_reranker : bool
        When True, apply a lightweight cross-encoder reranker on the candidate
        pool.  Adds ~100-200ms per query but significantly improves ranking.
        Off by default to keep inference fast.
    """
    import numpy as np

    index = bundle["index"]
    chunks = bundle["chunks"]
    model_name = bundle["manifest"]["model_name"]

    # --- Build or reuse BM25 retriever ---
    bm25 = None
    if hybrid:
        cache_key = ("bm25", id(chunks))
        if cache_key not in _model_cache:
            _model_cache[cache_key] = BM25SparseRetriever(chunks)
        bm25 = _model_cache[cache_key]

    # --- Dense retrieval (FAISS) — single original question only ---
    model = _ensure_query_model(model_name)
    q_vec = embed_texts(
        [question], model=model, model_name=model_name, batch_size=1
    )
    pool_size = min(candidate_pool, index.ntotal)
    scores, ids = index.search(q_vec, pool_size)
    dense_scores: dict[int, float] = {}
    for pos, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        dense_scores[idx] = float(scores[0][pos])

    # --- BM25 retrieval — use expanded variants for better recall ---
    bm25_scores: dict[int, float] = {}
    if bm25 is not None:
        variants = expand_query_variants(question)
        for variant in variants:
            for idx, score in bm25.search(variant, top_k=pool_size):
                if idx not in bm25_scores or score > bm25_scores[idx]:
                    bm25_scores[idx] = score
        # Min-max normalise BM25 scores into ~[0,1]
        if bm25_scores:
            vals = list(bm25_scores.values())
            bmin, bmax = min(vals), max(vals)
            if bmax > bmin:
                bm25_scores = {
                    k: (v - bmin) / (bmax - bmin) for k, v in bm25_scores.items()
                }

    # --- Fuse dense + BM25 ---
    # Dense scores are cosine-similarity (~0 to ~1).  BM25 scores are
    # min-max normalised to [0,1].  Linear combination with a small BM25
    # weight fixes exact-term misses without distorting semantic ranking.
    fused: list[tuple[int, float]] = []
    all_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
    for idx in all_ids:
        d_score = dense_scores.get(idx, 0.0)
        if hybrid and bm25 is not None:
            b_score = bm25_scores.get(idx, 0.0)
            combined = (1 - bm25_weight) * d_score + bm25_weight * b_score
        else:
            combined = d_score
        fused.append((idx, combined))

    fused.sort(key=lambda x: x[1], reverse=True)

    # --- Optional cross-encoder rerank on top candidates ---
    if use_reranker:
        fused = _rerank_candidates(question, chunks, fused[:candidate_pool])

    # --- Build final hits ---
    hits: list[dict] = []
    for idx, score in fused[:top_k]:
        chunk = chunks[idx]
        hits.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": round(score, 4),
            }
        )
    return hits


def _rerank_candidates(
    question: str,
    chunks: list[dict],
    candidates: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Lightweight cross-encoder rerank of a candidate pool.

    Uses ``ms-marco-MiniLM-L-6-v2`` (~80 MB) — adds ~100 ms per query but
    dramatically improves ranking quality.  The model is cached after first load.
    """
    cache_key = ("cross_encoder",)
    if cache_key not in _model_cache:
        try:
            from sentence_transformers import CrossEncoder

            _model_cache[cache_key] = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        except ImportError:
            # sentence-transformers already required; this should not fail
            return candidates
        except Exception:
            return candidates

    reranker = _model_cache[cache_key]
    pairs = [(question, chunks[idx]["text"]) for idx, _ in candidates]
    try:
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
    except Exception:
        return candidates

    # Replace scores with cross-encoder scores
    reranked = [
        (candidates[i][0], float(ce_scores[i])) for i in range(len(candidates))
    ]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
    hybrid: bool = True,
    bm25_weight: float = 0.15,
    use_reranker: bool = False,
) -> list[dict]:
    """Load the saved FAISS index from *document* and return top-k hits.

    See :func:`search_bundle` for the *hybrid*, *bm25_weight*, and
    *use_reranker* parameter docs.
    """
    index_path = document.get("artifacts", {}).get("index")
    if index_path is None:
        raise ValueError("document record is missing 'artifacts.index'")

    index = load_faiss_index(index_path)
    bundle = {
        "index": index,
        "chunks": document["chunks"],
        "manifest": {
            "model_name": document.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        },
    }
    return search_bundle(
        question,
        bundle,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=history,
        hybrid=hybrid,
        bm25_weight=bm25_weight,
        use_reranker=use_reranker,
    )


def split_sentences(text: str) -> list[str]:
    """Split text into candidate answer sentences."""
    import re

    if not text:
        return []
    # Split on sentence-ending punctuation followed by space or line-start
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Return one short answer sentence with a page tag when possible.

    Picks the sentence from the top hit that shares the most keywords with
    the question, and appends ``[Page X]``.
    """
    if not hits:
        return (
            "The document does not provide enough information "
            "to answer this question."
        )

    q_tokens = keyword_set(question)
    best_sentence = ""
    best_page = hits[0]["page"]
    best_overlap = -1

    for hit in hits[:3]:
        for sent in split_sentences(hit["text"]):
            overlap = len(q_tokens & keyword_set(sent))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sent
                best_page = hit["page"]

    if not best_sentence:
        best_sentence = hits[0]["text"][:200].strip()
        best_page = hits[0]["page"]

    return f"{best_sentence} [Page {best_page}]"


# ---------------------------------------------------------------------------
# 6. Project-facing wrappers
# ---------------------------------------------------------------------------


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root=None,
) -> dict:
    """Build one server-side document record.

    Returns a dict suitable for ``documents[chat_id]`` storage, with
    pages, chunks, retrieval assets, and empty history.
    """
    root = Path(artifact_root) if artifact_root else Path("artifacts")

    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=root,
    )

    paths = bundle["paths"]
    manifest = bundle["manifest"]

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "history": [],
        "model_name": model_name,
        "model_source": resolve_model_source(model_name),
        "chunk_mode": chunk_mode,
        "chunk_size": len(bundle["chunks"]),
        "embedding_dim": manifest["embedding_dim"],
        "artifacts": {
            "chunks": str(paths["chunks"]),
            "embeddings": str(paths["embeddings"]),
            "index": str(paths["index"]),
            "manifest": str(paths["manifest"]),
        },
    }


def extract_citations(
    answer: str,
    hits: list[dict] | None = None,
) -> list[int]:
    """Extract numeric PDF page citations from an answer string and/or hits."""
    import re

    pages: set[int] = set()

    # 1. Parse [Page N] markers in the answer text
    for m in re.finditer(r"\[Page (\d+)\]", answer):
        pages.add(int(m.group(1)))

    # 2. Fall back to hit pages
    if not pages and hits:
        pages = {h["page"] for h in hits}

    return sorted(pages)


def build_sources(hits: list[dict]) -> list[dict]:
    """Convert retrieval hits into frontend-friendly source objects."""
    return [
        {
            "chunk_id": h["chunk_id"],
            "page": h["page"],
            "score": h.get("score", 0),
            "preview": h["text"][:200].replace("\n", " "),
        }
        for h in hits
    ]


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Build a grounded prompt with retrieved evidence and recent history."""
    parts: list[str] = []

    if history:
        parts.append("### Conversation history")
        for turn in history[-6:]:  # last 3 turns (6 messages)
            role = turn.get("role", "user")
            content = turn.get("content", turn.get("question", ""))
            parts.append(f"{role}: {content}")
        parts.append("")

    parts.append("### Retrieved evidence")
    for h in hits:
        parts.append(f"[Chunk {h['chunk_id']} | Page {h['page']}] {h['text']}")
    parts.append("")

    parts.append("### User question")
    parts.append(question)
    parts.append("")
    parts.append(
        "Answer the question using only the retrieved evidence above. "
        "Cite factual claims with [Page X]. "
        "If the answer is not in the evidence, say that the document "
        "does not provide enough information. "
        "Never invent a page number."
    )

    return "\n".join(parts)


def _call_llm_answer(prompt: str, answer_model: str = "tencent/hy3:free") -> str:
    """Call the LLM via OpenRouter to produce an answer from the grounded prompt."""
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", answer_model),
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions only from the supplied evidence. "
                    "Cite factual claims with [Page X]. "
                    "If the answer is not in the evidence, say that the "
                    "document does not provide enough information. "
                    "Never invent a page number."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "tencent/hy3:free",
    hybrid: bool = True,
    bm25_weight: float = 0.15,
    use_reranker: bool = False,
) -> dict:
    """Run retrieval + answer generation for one question.

    Returns ``answer``, ``citations``, and ``sources``.
    Falls back to local sentence extraction when the API key is missing.
    """
    hits = search_document(
        question,
        document,
        top_k=top_k,
        candidate_pool=candidate_pool,
        hybrid=hybrid,
        bm25_weight=bm25_weight,
        use_reranker=use_reranker,
    )

    try:
        prompt = build_grounded_user_prompt(question, hits)
        answer = _call_llm_answer(prompt, answer_model=answer_model)
    except (RuntimeError, Exception):
        # Fall back to local answer extraction
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


def append_history(
    document: dict,
    question: str,
    result: dict,
) -> list[dict]:
    """Append a user/assistant turn to the in-memory history and return it."""
    history: list[dict] = document.setdefault("history", [])
    history.append({"role": "user", "content": question})
    history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result.get("citations", []),
        }
    )
    return history


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "tencent/hy3:free",
    hybrid: bool = True,
    bm25_weight: float = 0.15,
    use_reranker: bool = False,
) -> dict:
    """Answer one question and update the in-memory history.

    Returns the ``answer_document`` result plus the updated ``history`` list.
    """
    result = answer_document(
        document,
        question,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
        hybrid=hybrid,
        bm25_weight=bm25_weight,
        use_reranker=use_reranker,
    )
    history = append_history(document, question, result)
    result["history"] = history
    return result


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "tencent/hy3:free",
    hybrid: bool = True,
    bm25_weight: float = 0.15,
    use_reranker: bool = False,
) -> dict:
    """Route-facing wrapper: retrieve evidence, answer, and update history.

    This is the single function the ``POST /chat`` route calls for each turn.
    """
    return answer_document_turn(
        document,
        message,
        top_k=top_k,
        candidate_pool=candidate_pool,
        answer_model=answer_model,
        hybrid=hybrid,
        bm25_weight=bm25_weight,
        use_reranker=use_reranker,
    )


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Read PDF bytes and return ``[{page, text}]`` records for upload routes."""
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        cleaned = clean_text(raw)
        if cleaned:
            records.append({"page": page_number, "text": cleaned})
    return records


# ---------------------------------------------------------------------------
# 7. Evaluation helpers
# ---------------------------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Normalize text for simple string-based scoring."""
    import re

    if not text:
        return ""
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation for fuzzy matching
    text = re.sub(r"[^\w\s]", "", text)
    return text


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return True if any of *answers* appear in *text* after normalization."""
    norm_text = normalize_for_match(text)
    for ans in answers:
        norm_ans = normalize_for_match(ans)
        if norm_ans and norm_ans in norm_text:
            return True
    return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
):
    """Run evaluation and return one row per question.

    Parameters
    ----------
    eval_set : list[dict]
        Each entry has ``pdf_name``, ``question``, and ``answers`` (list of
        acceptable gold answer strings).
    documents_by_name : dict[str, dict]
        Mapping from ``pdf_name`` to prepared document records.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for evaluate_questions")

    rows: list[dict] = []
    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        gold_answers = item["answers"]

        document = documents_by_name.get(pdf_name)
        if document is None:
            rows.append(
                {
                    "pdf_name": pdf_name,
                    "question": question,
                    "gold_answers": gold_answers,
                    "answer": "DOCUMENT NOT FOUND",
                    "pages": [],
                    "retrieval_hit": False,
                    "answer_hit": False,
                    "error": f"document '{pdf_name}' not in documents_by_name",
                }
            )
            continue

        hits = search_document(
            question,
            document,
            top_k=top_k,
            candidate_pool=candidate_pool,
            hybrid=True,
            bm25_weight=0.15,
        )
        answer = best_sentence_answer(question, hits)

        # retrieval_hit: at least one hit page appears to contain any answer
        retrieval_hit = any(
            contains_any_answer(h["text"], gold_answers) for h in hits
        )

        # answer_hit: the local answer string itself contains any answer
        answer_hit = contains_any_answer(answer, gold_answers)

        rows.append(
            {
                "pdf_name": pdf_name,
                "question": question,
                "gold_answers": gold_answers,
                "answer": answer,
                "pages": sorted({h["page"] for h in hits}),
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Appendix B -- Chroma collection (optional)
# ---------------------------------------------------------------------------


def ensure_artifact_dirs(artifact_root=None) -> dict[str, Path]:
    """Return (and create) all artifact folders including Chroma storage."""
    root = Path(artifact_root) if artifact_root else Path("artifacts")
    dirs = {
        "raw_pages": root / "raw_pages",
        "chunks": root / "chunks",
        "embeddings": root / "embeddings",
        "reports": root / "reports",
        "chroma": root / "chroma",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _require_chromadb():
    """Import ``chromadb`` or raise a clear ImportError."""
    try:
        import chromadb

        return chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for the Chroma appendix path. "
            "Install it with: pip install chromadb"
        )


def build_chroma_collection(
    document_id: str,
    chunks: list[dict],
    embeddings: "np.ndarray",  # noqa: F821
    persist_dir,
) -> dict:
    """Create or reopen one Chroma collection for *document_id*."""
    import numpy as np

    chromadb = _require_chromadb()

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_name = f"rag_{document_id}"

    # Remove existing collection with the same name if present
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = [f"chunk-{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]
    emb_list = np.asarray(embeddings, dtype=np.float32).tolist()

    collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=emb_list
    )

    return {
        "collection_name": collection_name,
        "item_count": collection.count(),
        "persist_dir": str(persist_dir),
    }


def query_chroma_collection(
    document_id: str,
    query_embedding: "np.ndarray",  # noqa: F821
    persist_dir,
    top_k: int = 3,
) -> list[dict]:
    """Query one Chroma collection and return top-k hits."""
    import numpy as np

    chromadb = _require_chromadb()

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_name = f"rag_{document_id}"
    collection = client.get_collection(name=collection_name)

    q_vec = np.asarray(query_embedding, dtype=np.float32)
    if q_vec.ndim == 2:
        q_vec = q_vec[0]
    results = collection.query(
        query_embeddings=[q_vec.tolist()], n_results=top_k
    )

    hits: list[dict] = []
    ids_list = results.get("ids", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for i, chunk_id in enumerate(ids_list):
        meta = metas_list[i] if i < len(metas_list) else {}
        text = docs_list[i] if i < len(docs_list) else ""
        score = float(dists_list[i]) if i < len(dists_list) else 0.0
        hits.append(
            {
                "chunk_id": meta.get("chunk_id", chunk_id),
                "page": meta.get("page", 0),
                "text": text,
                "score": round(score, 4),
            }
        )
    return hits


def search_document_with_chroma(
    question: str,
    document: dict,
    persist_dir,
    top_k: int = 3,
    batch_size: int = 1,
) -> list[dict]:
    """Search the Chroma collection for a question."""
    model_name = document.get(
        "model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    q_vec = embed_texts(
        [question], model_name=model_name, batch_size=batch_size
    )
    return query_chroma_collection(
        document["document_id"], q_vec, persist_dir, top_k=top_k
    )


def answer_document_with_chroma(
    document: dict,
    question: str,
    persist_dir,
    top_k: int = 3,
    answer_model: str = "tencent/hy3:free",
) -> dict:
    """Answer using the Chroma collection instead of FAISS."""
    hits = search_document_with_chroma(
        question, document, persist_dir, top_k=top_k
    )

    try:
        prompt = build_grounded_user_prompt(question, hits)
        answer = _call_llm_answer(prompt, answer_model=answer_model)
    except (RuntimeError, Exception):
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }
