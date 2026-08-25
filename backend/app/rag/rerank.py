# from functools import lru_cache

# from sentence_transformers import CrossEncoder

# from app.config import Settings
# from app.schemas import SourceChunk


# DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# def semantic_rerank(
#     query: str,
#     chunks: list[SourceChunk],
#     settings: Settings,
# ) -> list[SourceChunk]:
#     """
#     Reranks RRF-fused candidate chunks using a dedicated semantic reranking model.

#     Input:
#     - query: rewritten standalone query
#     - chunks: RRF-fused candidates from hybrid_search()
#     - settings: app config

#     Flow:
#     1. Pair the query with each candidate chunk.
#     2. Score every query+chunk pair with a cross-encoder reranker.
#     3. Sort chunks by semantic relevance score.
#     4. Return final top K chunks for the LLM.
#     """

#     if not chunks:
#         return []

#     reranker = get_reranker()

#     pairs = [
#         [query, chunk.content]
#         for chunk in chunks
#     ]

#     scores = reranker.predict(pairs)

#     scored_chunks = [
#         (float(score), chunk)
#         for score, chunk in zip(scores, chunks)
#     ]

#     scored_chunks.sort(key=lambda item: item[0], reverse=True)

#     final_chunks: list[SourceChunk] = []

#     for score, chunk in scored_chunks[: settings.retrieval_k]:
#         chunk.score = score

#         existing_method = chunk.retrieval_method or "unknown"

#         if "semantic_rerank" not in existing_method:
#             chunk.retrieval_method = f"{existing_method}+semantic_rerank"

#         final_chunks.append(chunk)

#     return final_chunks


# @lru_cache
# def get_reranker() -> CrossEncoder:
#     """
#     Loads the reranker once and reuses it.

#     First actual call may download the model.
#     Later calls reuse the cached model in memory.
#     """

#     return CrossEncoder(DEFAULT_RERANK_MODEL)

