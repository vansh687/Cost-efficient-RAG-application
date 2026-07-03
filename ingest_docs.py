import os
import sys

# Add projectsyn1 to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.document_loader import DocumentLoader
from app.services.chunker import Chunker
from app.services.vector_db import VectorDBService

def main():
    db = VectorDBService()
    
    files = ["loader_specs.md", "db_specs.md"]
    for f in files:
        if not os.path.exists(f):
            print(f"Skipping {f}, file does not exist.")
            continue
        print(f"Loading {f}...")
        docs = DocumentLoader.load_document(f)
        
        print("Chunking...")
        chunker = Chunker()
        chunks = chunker.split_documents(docs)
        
        print("Ingesting to ChromaDB...")
        res = db.ingest_chunks(chunks)
        print(f"Ingestion results for {f}: Ingested: {res['inserted']}, Skipped (Duplicate): {res['skipped']}")

if __name__ == "__main__":
    main()
