import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from app.config import settings
from app.services.embedding import EmbeddingProvider

class VectorDBService:
    def __init__(self, collection_name: str = "rag_documents"):
        self.embedding_provider = EmbeddingProvider()
        
        # Ensure vectors folder exists
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingests split chunks into ChromaDB.
        Performs IDEMPOTENT ingestion by skipping chunk hashes that already exist in DB.
        """
        if not chunks:
            return {"inserted": 0, "skipped": 0}
            
        inserted_count = 0
        skipped_count = 0
        
        # Collect existing hashes from collection metadata/documents
        existing_hashes = set()
        # Query existing metadata to determine duplication
        results = self.collection.get()
        if results and "metadatas" in results and results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "hash" in meta:
                    existing_hashes.add(meta["hash"])

        # Filters chunks
        to_insert_texts = []
        to_insert_metadatas = []
        to_insert_ids = []
        
        for chunk in chunks:
            chunk_hash = chunk["hash"]
            if chunk_hash in existing_hashes:
                skipped_count += 1
                continue
                
            to_insert_texts.append(chunk["text"])
            to_insert_metadatas.append(chunk["metadata"])
            to_insert_ids.append(chunk_hash) # Using hash as unique ID
            existing_hashes.add(chunk_hash) # Avoid internal duplicates in current batch
            
        if to_insert_texts:
            embeddings = self.embedding_provider.get_embeddings(to_insert_texts)
            self.collection.add(
                documents=to_insert_texts,
                embeddings=embeddings,
                metadatas=to_insert_metadatas,
                ids=to_insert_ids
            )
            inserted_count = len(to_insert_texts)
            
        return {
            "inserted": inserted_count,
            "skipped": skipped_count
        }

    def retrieve(self, query: str, top_k: int = None, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Embeds the query and queries ChromaDB. Returns list of document dicts matching top_k.
        """
        top_k = top_k or settings.TOP_K
        query_vector = self.embedding_provider.get_embedding(query)
        
        # Construct optional where clauses
        where = None
        if metadata_filter:
            # Format metadata filters for Chroma (simple dict or complex logical expressions)
            where = metadata_filter

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where
        )
        
        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)
            
            for doc, meta, dist, id_ in zip(docs, metadatas, distances, ids):
                # Chroma distance cosine distance: 0 is exact match, 1 is orthogonal, 2 is opposite
                # Calculate similarity score: 1 - dist (for cosine)
                similarity = 1.0 - dist
                
                # Check similarity threshold if defined
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    retrieved_docs.append({
                        "id": id_,
                        "text": doc,
                        "metadata": meta,
                        "similarity": float(similarity)
                    })
                    
        return retrieved_docs
        
    def reset_db(self):
        """Helper to clear database collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
