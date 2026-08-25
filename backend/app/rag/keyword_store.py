import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.rag.vector_store import get_vector_store
from app.schemas import SourceChunk


def bm25_search(
    query: str,
    settings: Settings,
) -> list[SourceChunk]:
    """
    Runs BM25 keyword search over chunks stored in ChromaDB.

    Flow:
    1. Load all stored chunks from ChromaDB.
    2. Tokenize chunk text.
    3. Build BM25 index in memory.
    4. Score chunks against the query.
    5. Return top keyword matches.
    """

    documents = _load_all_chunks(settings)

    if not documents:
        return []

    tokenized_corpus = [
        _tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    top_results = [
        (doc, float(score))
        for doc, score in ranked[: settings.retrieval_k]
        if score > 0
    ]

    return [
        _source_from_document(
            document=doc,
            score=score,
            retrieval_method="bm25",
        )
        for doc, score in top_results
    ]


def _load_all_chunks(settings: Settings) -> list[Document]:
    """
    Loads all chunks from ChromaDB.

    BM25 needs the text corpus, so we retrieve stored documents and metadata.
    """

    vector_store = get_vector_store(settings)

    raw = vector_store.get(include=["documents", "metadatas"])

    documents: list[Document] = []

    raw_documents = raw.get("documents", [])
    raw_metadatas = raw.get("metadatas", [])

    for content, metadata in zip(raw_documents, raw_metadatas):
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {},
            )
        )

    return documents


def _source_from_document(
    document: Document,
    score: float | None,
    retrieval_method: str,
) -> SourceChunk:
    page = document.metadata.get("page")

    return SourceChunk(
        source_id=str(document.metadata.get("source_id", "unknown")),
        filename=str(document.metadata.get("filename", "unknown")),
        page=int(page) + 1 if isinstance(page, int) else None,
        chunk_index=document.metadata.get("chunk_index"),
        content=document.page_content,
        score=score,
        retrieval_method=retrieval_method,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())
