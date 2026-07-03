import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    
    LLM_PROVIDER: str = "mockup" # options: gemini, openai, mockup
    EMBEDDING_PROVIDER: str = "mockup" # options: gemini, openai, mockup
    
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    VECTOR_DB_PATH: str = "./chroma_db"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
