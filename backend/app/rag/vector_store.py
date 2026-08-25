from fastapi import HTTPException, status
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import Settings
import chromadb


def require_openai_key(settings: Settings) -> None:
    if (
        not settings.openai_api_key
        or settings.openai_api_key == "your_openai_api_key_here"
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is missing. Set it in backend/.env.",
        )


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """
    Creates the embedding model.

    Embeddings convert text into numeric vectors.
    These vectors allow semantic search.
    """

    require_openai_key(settings)

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store(settings: Settings) -> Chroma:
    """
    Creates the ChromaDB vector store using cosine distance
    for semantic similarity search.

    ChromaDB stores:
    - text chunks
    - embeddings
    - metadata
    """

    # return Chroma(
    #     collection_name=settings.collection_name,
    #     persist_directory=str(settings.chroma_dir),
    #     embedding_function=get_embeddings(settings),
    #     collection_metadata={"hnsw:space": "cosine"},
    # )
    require_openai_key(settings)

    client = chromadb.CloudClient(
        tenant=settings.chroma_tenant,
        database=settings.chroma_database,
        api_key=settings.chroma_api_key,
    )

    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
    )