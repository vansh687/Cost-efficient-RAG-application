import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings

class Chunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits loaded documents into smaller chunks, computes SHA-256 hash for deduplication,
        and returns list of chunk dicts ready for ingestion.
        """
        chunks = []
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            splits = self.splitter.split_text(text)
            for idx, split in enumerate(splits):
                # Calculate sha-256 hash of the content to enforce idempotent ingestion
                chunk_hash = hashlib.sha256(split.encode("utf-8")).hexdigest()
                
                # Combine original metadata with chunking metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": idx,
                    "hash": chunk_hash
                })
                
                chunks.append({
                    "text": split,
                    "hash": chunk_hash,
                    "metadata": chunk_metadata
                })
        return chunks
