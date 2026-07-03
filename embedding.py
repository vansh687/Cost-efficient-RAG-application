import os
from typing import List
from app.config import settings

class EmbeddingProvider:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.EMBEDDING_PROVIDER
        
        # Setup Gemini
        if self.provider == "gemini":
            import google.generativeai as genai
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment variables")
            genai.configure(api_key=api_key)
            self.model_name = "models/embedding-001"
            self.dimensions = 768
            self.client = genai
        # Setup OpenAI
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "text-embedding-ada-002"
            self.dimensions = 1536
        # Setup Mockup (no api key required)
        else:
            self.provider = "mockup"
            self.model_name = "mockup-embedding-model"
            self.dimensions = 384

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "gemini":
            embeddings = []
            for text in texts:
                result = self.client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            return embeddings
            
        elif self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
            
        else:
            # Deterministic mockup embeddings based on character codes
            mocked = []
            for text in texts:
                emb = []
                # Include length and hash-like properties to make different chunks stand out
                h = sum(ord(c) * (idx + 1) for idx, c in enumerate(text))
                for i in range(self.dimensions):
                    val = (h * (i + 13)) % 1000 / 1000.0
                    emb.append(val)
                mocked.append(emb)
            return mocked
