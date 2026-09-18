import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # AI API keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Embedding and LLM models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gemini-3.1-flash-lite")
    
    # Database settings (Optional MongoDB Atlas)
    MONGO_URL: str = os.getenv("MONGO_URL", "")
    DB_NAME: str = os.getenv("DB_NAME", "rag_db")
    VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "vector_documents")
    # Uploads path
    
    UPLOADS_DIR: Path = BASE_DIR / "uploads"

settings = Settings()
