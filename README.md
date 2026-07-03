# Cost-Efficient RAG Application

This is a modular Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Gemini/OpenAI interfaces. It supports idempotent document ingestion, structured operations logging, detailed cost estimations, and automated quality metrics evaluation.

## Architecture Pipeline

```
Documents (PDF, HTML, MD) -> Loader -> Chunker (RecursiveCharacterTextSplitter) 
  -> SHA-256 Hash Check -> Embeddings -> ChromaDB (HNWS Cosine) -> Retrieval 
  -> Prompt Engineering (Context-Conformed) -> LLM Answer + References & Citations
```

---

## Getting Started

### 1. Requirements

Install local packages:
```bash
pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Create a `.env` file based on `.env.example`:
```env
LLM_PROVIDER=mockup  # options: gemini, openai, mockup
EMBEDDING_PROVIDER=mockup # options: gemini, openai, mockup
VECTOR_DB_PATH=./chroma_db
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=5
```

### 3. Run FastAPI Application

```bash
uvicorn app.main:app --reload
```
API Documentation will be accessible at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## API Endpoints

- `POST /ingest`: Accepts files (PDF, HTML, MD) and inserts them idempotently into the database.
- `POST /ask`: Takes a JSON request like `{"question": "..."}` and returns the generated response, citations, retrieved chunks, latency metrics, token count, and computed costs.
- `POST /reset`: Resets the stored collection contents.

---

## Evaluation Metrics

The system calculates key retrieval and generation metrics over a 15-question ground truth dataset:
1. **Recall@5**: Fraction of relevant chunks retrieved.
2. **Hit Rate**: Probability that at least one relevant document is returned in Top-K.
3. **MRR (Mean Reciprocal Rank)**: Position of the first relevant chunk.
4. **nDCG@5 (Normalized Discounted Cumulative Gain)**: Ranking quality score.
5. **Faithfulness**: Factuality of LLM response relative to retrieved context.
6. **Answer Relevance**: Semantic alignment of LLM response to query.

---

## Cost Analysis: Local ChromaDB vs. Managed Vector DBs

Comparing cost structures for storing and querying vectors across scale dimensions:

### Assumptions:
- **Embedding Model**: `models/embedding-001` (768 dimensions), 1 embedding = 3.072 KB of vector data.
- **Operations**: 1 Ingestion write and 10 Query reads per vector per month.
- **Managed DBs (e.g., Pinecone Standard, Qdrant Cloud)**: Average cost is ~$0.10 per GB of memory storage + $20/month base clusters + query unit costs ($0.0001 per 1K queries).
- **Local ChromaDB**: Running on standard self-hosted VM instances (e.g., AWS EC2 t3.medium at ~$30/month) or local developer servers.

### Pricing Comparison Matrix

| Scale (Vectors) | Data Size | Local ChromaDB Cost/mo (Self-Hosted Instance) | Managed Vector DB Cost/mo (Standard Tier) | Trade-offs & Recommendations |
| :--- | :--- | :--- | :--- | :--- |
| **100K** | ~300 MB | **$0.00** (Runs locally or on free VM tiers) | **~$20.00 - $30.00** (Minimum cluster sizes apply) | **Local ChromaDB** wins for small-scale apps and prototyping due to zero additional licensing or cluster overhead. |
| **1M** | ~3 GB | **~$15.00** (e.g., basic VPS or AWS EC2 micro) | **~$45.00** | **Local ChromaDB** remains highly cost-efficient if low latency limits are tolerable. |
| **10M** | ~30 GB | **~$60.00** (Needs mid-tier VM with SSD storage) | **~$150.00 - $250.00** (Auto-scaled shard tiers) | **Managed DBs** win here. Although hardware costs are lower for local deployment, managed databases offer superior multi-node indexing, failover, and sub-millisecond querying SLAs. |
