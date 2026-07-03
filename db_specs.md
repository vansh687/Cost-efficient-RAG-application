# Database Specifications

Persistent storage is preferred to avoid rebuilding embeddings every time the FastAPI service restarts, saving time and API costs.

The ChromaDB collection is configured with 'hnsw:space' set to 'cosine' to compute cosine similarity scores between the query and chunk vectors.

Retrieval queries support filtering chunks by metadata keys such as 'source' file name and document 'type'.

A similarity threshold of 0.3 is applied so that chunks with low cosine similarity to the query are automatically filtered out.
