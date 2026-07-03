import time
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.config import settings
from app.services.document_loader import DocumentLoader
from app.services.chunker import Chunker
from app.services.vector_db import VectorDBService
from app.services.llm_generator import LLMGenerator
from app.utils.logger import log_operation
from app.utils.cost_calculator import CostCalculator

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Make sure temporary and storage dirs exist
os.makedirs("temp", exist_ok=True)

app = FastAPI(title="Cost-Efficient RAG Application", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup services
vector_db_service = VectorDBService()
llm_generator = LLMGenerator()

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback path if run from root directory
    with open("app/ui.html", "r", encoding="utf-8") as f:
        return f.read()

class IngestResponse(BaseModel):
    message: str
    inserted_chunks: int
    skipped_chunks: int
    latency_ms: float
    estimated_cost_usd: float

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    latency_metrics: Dict[str, float]
    token_usage: Dict[str, int]
    estimated_cost_usd: float

@app.get("/")
def read_root():
    return {
        "status": "online",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "chunk_size": settings.CHUNK_SIZE
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    start_time = time.time()
    temp_path = os.path.join("temp", file.filename)
    
    try:
        # Save upload to temp file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load document
        loader_start = time.time()
        docs = DocumentLoader.load_document(temp_path)
        loader_time = (time.time() - loader_start) * 1000
        
        # 2. Split chunks
        chunker = Chunker()
        chunks = chunker.split_documents(docs)
        
        # 3. Store in vector database
        db_start = time.time()
        ingest_result = vector_db_service.ingest_chunks(chunks)
        db_time = (time.time() - db_start) * 1000
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate embedding cost
        total_chars = sum(len(c["text"]) for c in chunks)
        # Approximate tokens (roughly 1 token per 4 chars)
        approx_tokens = total_chars // 4
        est_cost = CostCalculator.calculate_embedding_cost(settings.EMBEDDING_PROVIDER, approx_tokens)
        
        log_operation("ingest", {
            "filename": file.filename,
            "chunks_count": len(chunks),
            "inserted": ingest_result["inserted"],
            "skipped": ingest_result["skipped"],
            "loader_latency_ms": loader_time,
            "db_latency_ms": db_time,
            "total_latency_ms": latency_ms,
            "approx_tokens": approx_tokens,
            "estimated_cost_usd": est_cost
        })
        
        return IngestResponse(
            message="Document ingested successfully.",
            inserted_chunks=ingest_result["inserted"],
            skipped_chunks=ingest_result["skipped"],
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=round(est_cost, 6)
        )
        
    except Exception as e:
        log_operation("ingest_error", {"filename": file.filename, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    overall_start = time.time()
    
    try:
        # 1. Retrieve Context
        retrieval_start = time.time()
        chunks = vector_db_service.retrieve(
            query=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000
        
        # 2. Estimate Retrieval Embedding Cost
        query_tokens = len(request.question) // 4
        embedding_cost = CostCalculator.calculate_embedding_cost(settings.EMBEDDING_PROVIDER, query_tokens)
        
        # 3. Prompt + Generate Answer
        gen_start = time.time()
        answer, citations, token_usage = llm_generator.generate_answer(
            query=request.question,
            context_chunks=chunks
        )
        gen_latency = (time.time() - gen_start) * 1000
        
        # 4. Estimate Generation Cost
        generation_cost = CostCalculator.calculate_generation_cost(
            settings.LLM_PROVIDER,
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"]
        )
        
        total_cost = embedding_cost + generation_cost
        overall_latency = (time.time() - overall_start) * 1000
        
        latency_metrics = {
            "retrieval_ms": round(retrieval_latency, 2),
            "generation_ms": round(gen_latency, 2),
            "total_ms": round(overall_latency, 2)
        }
        
        log_operation("ask", {
            "query": request.question,
            "retrieved_chunk_count": len(chunks),
            "answer_length": len(answer),
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": gen_latency,
            "total_latency_ms": overall_latency,
            "estimated_cost_usd": total_cost
        })
        
        return AskResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=[{
                "id": c["id"],
                "text": c["text"],
                "similarity": c["similarity"],
                "metadata": c["metadata"]
            } for c in chunks],
            latency_metrics=latency_metrics,
            token_usage=token_usage,
            estimated_cost_usd=round(total_cost, 6)
        )
        
    except Exception as e:
        log_operation("ask_error", {"query": request.question, "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_database():
    try:
        vector_db_service.reset_db()
        return {"message": "Database collection has been reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
