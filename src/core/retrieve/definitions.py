from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import (
    definitions_index_dir,
    definitions_path,
    rag_embedding_batch_size,
    rag_embedding_model,
)
from src.core.rag.index.build_definitions_index import build_aliases
from src.core.rag.index.embeddings import E5Embeddings
from src.core.retrieve.retrieve import l2_score_to_cosine_similarity, load_vectorstore

GENERIC_LITERAL_TERMS = {
    "Работник",
    "Работодатель",
    "Трудовые отношения",
    "Трудовой договор",
}


def load_definitions_faiss() -> FAISS | None:
    if not (definitions_index_dir / "index.faiss").exists():
        return None
    embeddings = E5Embeddings(rag_embedding_model, batch_size=rag_embedding_batch_size)
    return load_vectorstore(definitions_index_dir.as_posix(), embeddings)


@lru_cache(maxsize=1)
def load_definitions(path: str | Path = definitions_path.as_posix()) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_definition_chunk_ids(path: str | Path = definitions_path.as_posix()) -> frozenset[str]:
    definitions = load_definitions(path)
    chunk_ids: set[str] = set()
    for payload in definitions.values():
        chunk_ids.update(str(chunk_id) for chunk_id in payload.get("chunks") or [])
    return frozenset(chunk_ids)


def retrieve_top_definitions(
    query_text: str,
    faiss: FAISS,
    top_k: int = 5,
) -> list[dict]:
    search_data = faiss.similarity_search_with_score(query_text, k=top_k)
    result = []
    for doc, score in search_data:
        meta_data = doc.metadata.copy()
        meta_data["embedding_text"] = doc.page_content
        result.append(
            {
                "meta_data": meta_data,
                "similarity": l2_score_to_cosine_similarity(score),
                "distance": float(score),
            }
        )
    return result


def literal_definition_matches(query_text: str, definitions: dict) -> list[dict]:
    matches = []

    for term, payload in definitions.items():
        if term in GENERIC_LITERAL_TERMS:
            continue

        aliases = build_aliases(term)
        matched_alias = next(
            (
                alias
                for alias in aliases
                if re.search(
                    rf"(?<![0-9A-Za-zА-Яа-яЁё]){re.escape(alias)}(?![0-9A-Za-zА-Яа-яЁё])",
                    query_text,
                    flags=re.IGNORECASE,
                )
            ),
            None,
        )
        if matched_alias is None:
            continue

        matches.append(
            {
                "meta_data": {
                    "definition_id": f"def:{term}",
                    "doc_type": "definition",
                    "term": term,
                    "aliases": aliases,
                    "source_chunk_ids": payload.get("chunks") or [],
                    "definition_text": (payload.get("text") or "").strip(),
                    "matched_alias": matched_alias,
                },
                "similarity": 1.0,
                "match_type": "literal",
            }
        )

    return matches
