import os
import pytest
from app.services.document_loader import DocumentLoader
from app.services.chunker import Chunker
from app.services.vector_db import VectorDBService
from app.services.llm_generator import LLMGenerator
from app.evaluation.evaluator import RAGEvaluator

@pytest.fixture
def temp_markdown_file():
    path = "temp_test_doc.md"
    content = (
        "# Testing Doc\n\n"
        "The primary role of the loader is parsing diverse document formats like PDF, HTML, and Markdown, "
        "ensuring that relevant text along with accurate metadata is correctly parsed for downstream chunking.\n\n"
        "By default, the chunk size is set to 800 characters and the chunk overlap is set to 150 characters."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_document_loader(temp_markdown_file):
    docs = DocumentLoader.load_document(temp_markdown_file)
    assert len(docs) == 1
    assert "Testing Doc" in docs[0]["text"]
    assert docs[0]["metadata"]["source"] == "temp_test_doc.md"
    assert docs[0]["metadata"]["type"] == "markdown"

def test_chunker_and_deduplication(temp_markdown_file):
    docs = DocumentLoader.load_document(temp_markdown_file)
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split_documents(docs)
    
    assert len(chunks) > 1
    # Check that SHA-256 is present and structured
    for c in chunks:
        assert "hash" in c
        assert len(c["hash"]) == 64
        
def test_vector_db_operations(temp_markdown_file):
    db = VectorDBService(collection_name="test_collection")
    db.reset_db()
    
    docs = DocumentLoader.load_document(temp_markdown_file)
    chunker = Chunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split_documents(docs)
    
    # Check Ingestion
    res1 = db.ingest_chunks(chunks)
    assert res1["inserted"] == len(chunks)
    assert res1["skipped"] == 0
    
    # Check Idempotency (duplicates should be skipped)
    res2 = db.ingest_chunks(chunks)
    assert res2["inserted"] == 0
    assert res2["skipped"] == len(chunks)
    
    # Check Retrieval
    retrieved = db.retrieve(query="primary role of the loader", top_k=2)
    assert len(retrieved) > 0
    assert any("loader" in c["text"].lower() for c in retrieved)

def test_llm_generator():
    gen = LLMGenerator(provider="mockup")
    context = [
        {"id": "hash1", "text": "The primary role of the loader is parsing diverse document formats.", "metadata": {"source": "doc1.md"}}
    ]
    ans, citations, tokens = gen.generate_answer("What is the primary role of the loader?", context)
    assert len(citations) > 0
    assert "Doc ID: 1" in ans
    assert tokens["prompt_tokens"] > 0
    
def test_evaluator_metrics():
    retrieved = ["This is context one.", "This is context two."]
    gt = ["This is context one."]
    
    recall = RAGEvaluator.compute_recall_at_k(retrieved, gt, k=2)
    assert recall == 1.0
    
    mrr = RAGEvaluator.compute_mrr(retrieved, gt)
    assert mrr == 1.0
    
    faithfulness = RAGEvaluator.evaluate_faithfulness("This is context one [Doc ID: 1].", retrieved)
    assert faithfulness == 1.0
