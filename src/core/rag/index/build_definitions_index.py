from __future__ import annotations

import json
import re

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    definitions_index_dir,
    definitions_path,
    rag_embedding_batch_size,
    rag_embedding_model,
)
from src.core.rag.index.embeddings import E5Embeddings


def build_definitions_index() -> None:
    definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
    docs: list[Document] = []

    for term, payload in definitions.items():
        text = (payload.get("text") or "").strip()
        if not text:
            continue

        chunks = payload.get("chunks") or []
        aliases = build_aliases(term)
        source_articles = sorted(
            {
                match.group(1)
                for chunk_id in chunks
                if (match := re.match(r"art:([^:]+):", chunk_id))
            }
        )
        embedding_text = (
            f"термин: {term}\n"
            f"синонимы: {', '.join(aliases)}\n"
            f"определение: {text}"
        )
        docs.append(
            Document(
                page_content=embedding_text,
                metadata={
                    "definition_id": f"def:{term}",
                    "doc_type": "definition",
                    "term": term,
                    "aliases": aliases,
                    "source_chunk_ids": chunks,
                    "source_articles": source_articles,
                    "definition_text": text,
                },
            )
        )

    embeddings = E5Embeddings(rag_embedding_model, batch_size=rag_embedding_batch_size)
    vs = FAISS.from_documents(docs, embeddings)

    definitions_index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(definitions_index_dir.as_posix())


def build_aliases(term: str) -> list[str]:
    aliases = {term.strip()}
    for part in re.split(r"\s*/\s*", term):
        part = part.strip()
        if part:
            aliases.add(part)

    cleaned_aliases = set()
    for alias in aliases:
        cleaned_aliases.add(alias)
        cleaned_aliases.add(re.sub(r"\s*\([^)]*\)", "", alias).strip())

    return sorted(alias for alias in cleaned_aliases if alias)


if __name__ == "__main__":
    build_definitions_index()
