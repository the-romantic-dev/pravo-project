from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

from transformers import TextClassificationPipeline
from sentence_transformers import CrossEncoder

from src import config
from src.core.classification.contradiction import contradiction_score, get_nli_pipeline
from src.core.pdf.pdf_chunks_document import PdfChunk, pdf_to_chunks_document
from src.core.retrieve.definitions import (
    literal_definition_matches,
    load_definition_chunk_ids,
    load_definitions,
    load_definitions_faiss,
    retrieve_top_definitions,
)
from src.core.retrieve.retrieve import load_fiass, retrieve_top_k
from src.core.retrieve.rerank import load_reranker, rerank_top_k

DEFAULT_CONTRADICTION_MODEL = "cointegrated/rubert-base-cased-nli-threeway"
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
DEFAULT_DEFINITION_TOP_K = 4
DEFAULT_DEFINITION_SIMILARITY_THRESHOLD = 0.72
DEFAULT_RUN_RERANKING = True
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE = 16
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_RERANKER_CANDIDATE_MULTIPLIER = 10
ANALYSIS_SCHEMA_VERSION = 3
BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class MatchResult:
    match_id: str
    similarity: float
    norm_text: str
    hierarchy_path: list[str]
    article_number: str
    part_number: str
    subpart_number: str
    rerank_score: float | None
    contradiction_score: float | None
    auto_label: str


@dataclass(slots=True)
class DefinitionHit:
    term: str
    text: str
    chunks: list[str]
    source_articles: list[str]
    similarity: float | None
    match_type: str


@dataclass(slots=True)
class ClauseResult:
    clause_id: str
    section: str
    point_number: str | None
    text: str
    page_to_bbox: dict[int, BBox]
    definitions: list[DefinitionHit]
    matches: list[MatchResult]


@dataclass(slots=True)
class AnalysisResult:
    analysis_id: str
    created_at: str
    pdf_name: str
    parameters: dict
    clauses: list[ClauseResult]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["clauses"] = [
            {
                "clause_id": clause.clause_id,
                "section": clause.section,
                "point_number": clause.point_number,
                "text": clause.text,
                "page_to_bbox": clause.page_to_bbox,
                "definitions": [asdict(definition) for definition in clause.definitions],
                "matches": [asdict(match) for match in clause.matches],
            }
            for clause in self.clauses
        ]
        return payload


@lru_cache(maxsize=1)
def get_cached_faiss():
    return load_fiass()


@lru_cache(maxsize=1)
def get_cached_definitions_faiss():
    return load_definitions_faiss()


@lru_cache(maxsize=1)
def get_cached_definitions():
    return load_definitions()


@lru_cache(maxsize=1)
def get_cached_definition_chunk_ids():
    return load_definition_chunk_ids()


@lru_cache(maxsize=1)
def get_cached_nli() -> TextClassificationPipeline:
    return get_nli_pipeline()


@lru_cache(maxsize=1)
def get_cached_reranker() -> CrossEncoder:
    return load_reranker(
        get_reranker_model_name(),
        max_length=get_reranker_max_length(),
    )


def get_reranker_model_name() -> str:
    return str(getattr(config, "reranker_model", DEFAULT_RERANKER_MODEL))


def get_reranker_batch_size() -> int:
    return int(getattr(config, "reranker_batch_size", DEFAULT_RERANKER_BATCH_SIZE))


def get_reranker_max_length() -> int:
    return int(getattr(config, "reranker_max_length", DEFAULT_RERANKER_MAX_LENGTH))


def get_reranker_candidate_multiplier() -> int:
    return int(
        getattr(
            config,
            "reranker_candidate_multiplier",
            DEFAULT_RERANKER_CANDIDATE_MULTIPLIER,
        )
    )


def analyze_pdf(
    pdf_path: Path,
    top_k: int = 3,
    definition_top_k: int = DEFAULT_DEFINITION_TOP_K,
    definition_similarity_threshold: float = DEFAULT_DEFINITION_SIMILARITY_THRESHOLD,
    max_clauses: int | None = 40,
    contradiction_model_name: str = DEFAULT_CONTRADICTION_MODEL,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    run_contradiction_scoring: bool = True,
    run_reranking: bool = DEFAULT_RUN_RERANKING,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AnalysisResult:
    clauses = extract_clauses(pdf_path, max_clauses=max_clauses)
    analysis_clauses: list[ClauseResult] = []
    total = len(clauses)
    faiss = get_cached_faiss()
    definitions_faiss = get_cached_definitions_faiss() if definition_top_k > 0 else None
    definitions = get_cached_definitions() if definition_top_k > 0 else {}
    definition_chunk_ids = get_cached_definition_chunk_ids()
    nli = get_cached_nli() if run_contradiction_scoring else None
    reranker = get_cached_reranker() if run_reranking else None

    for idx, clause_data in enumerate(clauses, start=1):
        if progress_callback is not None:
            progress_callback(idx - 1, total, f"Анализ пункта {idx}/{total}")

        clause_id = f"clause-{idx}"
        query_text = clause_data["text"]
        definition_hits = find_definition_hits(
            query_text=query_text,
            definitions_faiss=definitions_faiss,
            definitions=definitions,
            top_k=definition_top_k,
            similarity_threshold=definition_similarity_threshold,
        )
        matches = find_matches(
            clause_id=clause_id,
            query_text=query_text,
            faiss=faiss,
            nli=nli,
            top_k=top_k,
            contradiction_model_name=contradiction_model_name,
            contradiction_threshold=contradiction_threshold,
            run_contradiction_scoring=run_contradiction_scoring,
            reranker=reranker,
            run_reranking=run_reranking,
            excluded_chunk_ids=definition_chunk_ids,
        )
        analysis_clauses.append(
            ClauseResult(
                clause_id=clause_id,
                section=clause_data["section"],
                point_number=clause_data["point_number"],
                text=query_text,
                page_to_bbox=clause_data["page_to_bbox"],
                definitions=definition_hits,
                matches=matches,
            )
        )

    if progress_callback is not None:
        progress_callback(total, total, "Анализ завершен")

    return AnalysisResult(
        analysis_id=f"analysis-{uuid4().hex[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        pdf_name=pdf_path.name,
        parameters={
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "top_k": top_k,
            "definition_top_k": definition_top_k,
            "definition_similarity_threshold": definition_similarity_threshold,
            "max_clauses": max_clauses,
            "contradiction_model_name": contradiction_model_name,
            "contradiction_threshold": contradiction_threshold,
            "run_contradiction_scoring": run_contradiction_scoring,
            "reranker_model": get_reranker_model_name() if run_reranking else None,
            "run_reranking": run_reranking,
        },
        clauses=analysis_clauses,
    )


def extract_clauses(pdf_path: Path, max_clauses: int | None = 40) -> list[dict]:
    chunks_document = pdf_to_chunks_document(pdf_path)
    clauses: list[dict] = []

    for chunk in chunks_document.chunks:
        if not chunk.text.strip():
            continue

        clauses.append(chunk_to_clause(chunk))
        if max_clauses is not None and len(clauses) >= max_clauses:
            return clauses

    return clauses


def chunk_to_clause(chunk: PdfChunk) -> dict:
    return {
        "section": chunk.heading,
        "point_number": chunk.point_number,
        "text": chunk.text.strip(),
        "page_to_bbox": chunk.page_to_bbox,
    }


def find_definition_hits(
    query_text: str,
    definitions_faiss,
    definitions: dict,
    top_k: int,
    similarity_threshold: float,
) -> list[DefinitionHit]:
    if top_k <= 0:
        return []

    candidates = literal_definition_matches(query_text, definitions)
    if definitions_faiss is not None:
        candidates.extend(
            item
            for item in retrieve_top_definitions(
                query_text=query_text,
                faiss=definitions_faiss,
                top_k=top_k,
            )
            if float(item.get("similarity", 0.0)) >= similarity_threshold
        )

    by_term: dict[str, DefinitionHit] = {}
    for item in candidates:
        meta_data = item.get("meta_data", {})
        term = str(meta_data.get("term") or "").strip()
        text = str(meta_data.get("definition_text") or "").strip()
        if not term or not text:
            continue

        chunks = [str(chunk_id) for chunk_id in meta_data.get("source_chunk_ids", [])]
        hit = DefinitionHit(
            term=term,
            text=text,
            chunks=chunks,
            source_articles=extract_articles_from_chunks(chunks),
            similarity=float(item.get("similarity", 0.0)),
            match_type=str(item.get("match_type") or "semantic"),
        )
        current = by_term.get(term)
        if current is None or definition_hit_rank(hit) > definition_hit_rank(current):
            by_term[term] = hit

    return sorted(
        by_term.values(),
        key=lambda hit: (
            hit.match_type != "literal",
            -(hit.similarity or 0.0),
            hit.term,
        ),
    )[:top_k]


def definition_hit_rank(hit: DefinitionHit) -> tuple[int, float]:
    return (1 if hit.match_type == "literal" else 0, hit.similarity or 0.0)


def extract_articles_from_chunks(chunks: list[str]) -> list[str]:
    articles = {
        match.group(1)
        for chunk_id in chunks
        if (match := re.match(r"art:([^:]+):", chunk_id))
    }
    return sorted(articles)


def find_matches(
    clause_id: str,
    query_text: str,
    faiss,
    nli: TextClassificationPipeline | None,
    top_k: int,
    contradiction_model_name: str,  # kept for response payload compatibility
    contradiction_threshold: float,
    run_contradiction_scoring: bool,
    reranker: CrossEncoder | None = None,
    run_reranking: bool = DEFAULT_RUN_RERANKING,
    excluded_chunk_ids: set[str] | frozenset[str] | None = None,
) -> list[MatchResult]:
    excluded_chunk_ids = excluded_chunk_ids or set()
    reranker_candidate_multiplier = get_reranker_candidate_multiplier()
    candidate_count = (
        max(top_k, top_k * reranker_candidate_multiplier)
        if run_reranking and reranker is not None
        else max(top_k, top_k * 4)
    )
    retrieve_results = retrieve_top_k(
        query_text=query_text,
        faiss=faiss,
        top_k=candidate_count,
    )
    retrieve_results = [
        item
        for item in retrieve_results
        if str(item.get("meta_data", {}).get("chunk_id") or "") not in excluded_chunk_ids
    ]
    if run_reranking and reranker is not None:
        retrieve_results = rerank_top_k(
            query_text=query_text,
            retrieve_results=retrieve_results,
            reranker=reranker,
            top_k=top_k,
            batch_size=get_reranker_batch_size(),
        )
    matches: list[MatchResult] = []

    for item in retrieve_results:
        meta_data = item.get("meta_data", {})
        chunk_id = str(meta_data.get("chunk_id") or "")
        if chunk_id in excluded_chunk_ids:
            continue

        norm_text = (
            meta_data.get("original_text")
            or meta_data.get("normalized_text")
            or ""
        ).strip()
        score = None
        auto_label = "not_scored"

        if run_contradiction_scoring and norm_text and nli is not None:
            score = contradiction_score(
                nli,
                query_text,
                norm_text,
                bidirectional=True,
            )
            auto_label = (
                "contradiction"
                if score >= contradiction_threshold
                else "not_contradiction"
            )

        matches.append(
            MatchResult(
                match_id=f"{clause_id}-match-{len(matches) + 1}",
                similarity=float(item.get("similarity", 0.0)),
                norm_text=norm_text,
                hierarchy_path=list(meta_data.get("hierarchy_path", [])),
                article_number=str(meta_data.get("article_number", "")),
                part_number=str(meta_data.get("part_number", "")),
                subpart_number=str(meta_data.get("subpart_number", "")),
                rerank_score=(
                    float(item["rerank_score"])
                    if item.get("rerank_score") is not None
                    else None
                ),
                contradiction_score=score,
                auto_label=auto_label,
            )
        )
        if len(matches) >= top_k:
            break

    return matches


def save_review(
    output_path: Path,
    analysis: AnalysisResult,
    labels: dict[str, dict],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": analysis.to_dict(),
        "manual_labels": labels,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
