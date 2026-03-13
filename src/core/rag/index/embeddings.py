from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize

    def _embed(self, texts: List[str], prefix: str, show_progress_bar: bool = False) -> List[List[float]]:
        prefixed = [f"{prefix}{(t or '').strip()}" for t in texts]
        emb = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
        )
        if hasattr(emb, "tolist"):
            return emb.tolist()
        return [list(v) for v in emb]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "passage: ")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "query: ")[0]

