from langchain_core.documents import Document

from app.config import Settings
from app.rag.keyword_store import bm25_search
from app.rag.vector_store import get_vector_store
from app.schemas import SourceChunk
# from app.rag.rerank import semantic_rerank


RRF_K = 60


def vector_search(
    query: str,
    settings: Settings,
) -> list[SourceChunk]:
    """
    Runs semantic vector search against ChromaDB.

    This is useful for debugging because it returns vector scores.
    Final advanced RAG flow uses vector_search_mmr().
    """

    vector_store = get_vector_store(settings)

    results = vector_store.similarity_search_with_score(
        query=query,
        k=settings.retrieval_k,
    )

    return [
        _source_from_document(
            document=doc,
            score=score,
            retrieval_method="vector",
        )
        for doc, score in results
    ]


def vector_search_mmr(
    query: str,
    settings: Settings,
) -> list[SourceChunk]:
    """
    Runs MMR vector search.

    MMR returns chunks that are:
    - relevant to the query
    - diverse from each other
    """

    vector_store = get_vector_store(settings)

    docs = vector_store.max_marginal_relevance_search( #embed the query and retrieve the most relevant and diverse documents
        query=query,
        k=settings.retrieval_k,
        fetch_k=settings.retrieval_fetch_k,
    )

    return [
        _source_from_document(
            document=doc,
            score=None,
            retrieval_method="vector_mmr",
        )
        for doc in docs
    ]


def hybrid_search(
    query: str,
    settings: Settings,
) -> list[SourceChunk]:
    """
    Runs hybrid retrieval with RRF fusion.

    Flow:
    1. Run vector search using MMR.
    2. Run BM25 keyword search.
    3. Fuse both ranked lists using RRF.
    4. Return final top K chunks.
    """

    vector_results = vector_search_mmr(query, settings)
    keyword_results = bm25_search(query, settings)

    return rrf_fusion(
        result_lists=[vector_results, keyword_results],
        final_k=settings.retrieval_k * 2,
    )


def retrieve_final_chunks(
    query: str,
    settings: Settings,
) -> list[SourceChunk]:
    """
    Full retrieval pipeline.

    Flow:
    1. Vector search using MMR.
    2. BM25 keyword search.
    3. RRF fusion.
    4. Rerank hybrid candidates.
    5. Return final top K chunks.
    """
    return hybrid_search(query, settings)



    # If we want to add reranker we can use this 
    # candidates = hybrid_search(
    #     query=query,
    #     settings=settings,
    # )
    # return semantic_rerank(
    #     query=query,
    #     chunks=candidates,
    #     settings=settings,
    # )


def rrf_fusion(
    result_lists: list[list[SourceChunk]],
    final_k: int,
) -> list[SourceChunk]:
    """
    Combines multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score:
        1 / (RRF_K + rank)

    Rank starts at 1.

    RRF does not compare raw vector/BM25 scores.
    It only uses each chunk's rank position.
    """

    fused_scores: dict[str, float] = {}
    chunks_by_key: dict[str, SourceChunk] = {}
    methods_by_key: dict[str, set[str]] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = _chunk_key(chunk)

            fused_scores[key] = fused_scores.get(key, 0.0) + (
                1.0 / (RRF_K + rank)
            )
            chunks_by_key[key] = chunk

            method = chunk.retrieval_method or "unknown"
            methods_by_key.setdefault(key, set()).add(method)

    ranked_keys = sorted(
        fused_scores.keys(),
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    fused_chunks: list[SourceChunk] = []

    for key in ranked_keys[:final_k]:
        chunk = chunks_by_key[key]
        chunk.score = fused_scores[key]
        chunk.retrieval_method = "+".join(sorted(methods_by_key[key]))
        fused_chunks.append(chunk)

    return fused_chunks


def _source_from_document(
    document: Document,
    score: float | None,
    retrieval_method: str,
) -> SourceChunk:
    """
    Converts a LangChain Document into our API response schema.
    """

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


def _chunk_key(chunk: SourceChunk) -> str:
    """
    Creates a stable unique key for one chunk.

    This is how we detect the same chunk returned by both vector search and BM25.
    """

    return f"{chunk.source_id}:{chunk.chunk_index}"
