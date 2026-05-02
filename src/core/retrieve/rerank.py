from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from src.core.util.device import get_model_device


def load_reranker(
    model_name: str,
    max_length: int = 512,
    device: str | None = None,
) -> CrossEncoder:
    return CrossEncoder(model_name, max_length=max_length, device=device or get_model_device())


def rerank_top_k(
    query_text: str,
    retrieve_results: Sequence[dict],
    reranker: CrossEncoder,
    top_k: int,
    batch_size: int = 16,
) -> list[dict]:
    if top_k <= 0 or not retrieve_results:
        return []

    pairs = [
        (
            query_text,
            (
                item.get("meta_data", {}).get("original_text")
                or item.get("meta_data", {}).get("normalized_text")
                or ""
            ),
        )
        for item in retrieve_results
    ]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    ranked = []
    for index, (item, score) in enumerate(zip(retrieve_results, scores, strict=False)):
        reranked_item = item.copy()
        reranked_item["retrieval_similarity"] = float(item.get("similarity", 0.0))
        reranked_item["rerank_score"] = float(score)
        ranked.append((float(score), -index, reranked_item))

    return [
        item
        for _, _, item in sorted(
            ranked,
            key=lambda value: (value[0], value[1]),
            reverse=True,
        )
    ][:top_k]
