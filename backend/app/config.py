from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Loaded from .env
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str

    # RAG configuration loaded from .env
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    retrieval_fetch_k: int

    # Application defaults
    collection_name: str = "rag_documents"
    #upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    #chroma_dir: Path = BACKEND_DIR / "data" / "chroma"

    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str


# local storage
# @lru_cache
# def get_settings() -> Settings:
#     settings = Settings()

#     settings.upload_dir.mkdir(parents=True, exist_ok=True)
#     settings.chroma_dir.mkdir(parents=True, exist_ok=True)

#     return settings

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
