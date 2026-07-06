"""
reranker.py

Reranking layer for production RAG.

We reuse the same idea from our Adaptive-RAG implementation:

1. Retrieve more candidates than we need.
2. Score each candidate with a cross-encoder reranker.
3. Sort by reranker score.
4. Return the strongest chunks.

Embedding search is fast but approximate.
Reranking is slower but more precise.
"""

from sentence_transformers import CrossEncoder


_reranker_model = None


def get_reranker_model():
    """
    Load reranker once and reuse it.

    Model:
        BAAI/bge-reranker-base

    This is a cross-encoder:
        input = (query, document chunk)
        output = relevance score
    """

    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-base")

    return _reranker_model


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank retrieved chunks using BGE reranker.

    Args:
        query:
            User question.

        results:
            List of candidate result dictionaries.
            Each result must have a 'preview' field.

        top_k:
            Number of final reranked results to return.
    """

    if not results:
        return []

    model = get_reranker_model()

    pairs = []

    for item in results:
        pairs.append(
            [
                query,
                item.get("preview", ""),
            ]
        )

    scores = model.predict(pairs)

    reranked = []

    for item, score in zip(results, scores):
        new_item = dict(item)
        new_item["reranker_score"] = float(score)
        reranked.append(new_item)

    reranked = sorted(
        reranked,
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]