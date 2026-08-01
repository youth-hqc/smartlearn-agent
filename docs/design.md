# SmartLearn Agent - Product Design

## User Stories

1. As a **student**, I want to **upload a PDF and ask questions about its content**, so that **I can study course materials more efficiently instead of manually searching through slides**.
2. As a **student**, I want to **see page-number citations in every answer**, so that **I can quickly verify the answer against the original PDF and understand the full context**.
3. As a **student**, I want to **ask follow-up questions in a conversation**, so that **I can deepen my understanding of a topic without re-pasting the document each time**.

## Feature List

| Priority | Feature | Day | Notes |
|----------|---------|-----|-------|
| P0 | PDF text extraction | Day 2 | Foundation; nothing works without it |
| P0 | LLM Q&A with page citation | Day 2 | Core feature: upload a PDF, ask questions, get cited answers |
| P1 | RAG pipeline (chunk + embed + search) | Day 3 | Handles long PDFs by retrieving only the most relevant chunks |
| P1 | Web UI (React + FastAPI) | Day 3 | Lets users interact through a browser instead of the terminal |
| P2 | Chat history / multi-turn conversation | Day 3 | Remembers earlier questions and supports follow-ups |

## What We Will NOT Build

- **User authentication / login** — workshop time is limited; focus on core functionality
- **Multi-file support (uploading multiple PDFs at once)** — perfect the single-PDF experience first
- **Mobile app** — web version only; responsive design is out of scope for this workshop

## Data Flow

### Day 2: Simple Mode

```
PDF File
  -> [extract text]          # Use pdfplumber to pull text from each page
  -> pages[]                 # Array of page texts with page numbers
  -> [build prompt]          # Combine: system prompt + numbered pages + user question
  -> [LLM]                   # OpenRouter: qwen/qwen3.5-flash-02-23
  -> Answer with [Page X]    # LLM returns a cited response
```

### Day 3: RAG Mode

```
PDF -> [extract text] -> pages
     -> [split into chunks] -> chunks with source_page
     -> [embed] -> embeddings
     -> [vector store (FAISS)]   # storage

Question -> [encode] -> [similarity search] -> relevant chunks -> [LLM] -> Answer
```

### RAG in One Picture

```
Traditional: entire PDF text ──────────────────────> [LLM] -> answer
             (may be too long and exceed the AI input limit)

RAG:         entire PDF text -> chunk -> embed -> store
             user question -> embed -> search for the most relevant chunks -> [LLM] -> answer
             (sends only the most relevant parts; much more efficient)
```
