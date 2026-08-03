FROM python:3.11-slim

WORKDIR /app

# ---- System dependencies for faiss / sentence-transformers / openblas ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download the embedding model so the first request doesn't OOM ----
#     Railway free tier has 512 MB RAM; downloading + loading a model
#     at request-time often triggers an OOM kill → 502 Bad Gateway.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu') \
"

# ---- Application code ----
COPY . .

# ---- Runtime ----
#     HF_HOME makes the model cache discoverable inside the container.
ENV HF_HOME=/root/.cache/huggingface/hub

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
