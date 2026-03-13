from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path
from typing import Callable
from uuid import uuid4

from transformers import TextClassificationPipeline

from src.core.classification.contradiction import contradiction_score, get_nli_pipeline
from src.core.pdf.pdf_chunks_document import PdfChunk, pdf_to_chunks_document
from src.core.retrieve.retrieve import load_fiass, retrieve_top_k

DEFAULT_CONTRADICTION_MODEL = "cointegrated/rubert-base-cased-nli-threeway"
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
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
    contradiction_score: float | None
    auto_label: str


@dataclass(slots=True)
class ClauseResult:
    clause_id: str
    section: str
    point_number: str | None
    text: str
    page_to_bbox: dict[int, BBox]
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
                "matches": [asdict(match) for match in clause.matches],
            }
            for clause in self.clauses
        ]
        return payload


@lru_cache(maxsize=1)
def get_cached_faiss():
    return load_fiass()


@lru_cache(maxsize=1)
def get_cached_nli() -> TextClassificationPipeline:
    return get_nli_pipeline()


def analyze_pdf(
    pdf_path: Path,
    top_k: int = 3,
    max_clauses: int | None = 40,
    contradiction_model_name: str = DEFAULT_CONTRADICTION_MODEL,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
    run_contradiction_scoring: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> AnalysisResult:
    clauses = extract_clauses(pdf_path, max_clauses=max_clauses)
    analysis_clauses: list[ClauseResult] = []
    total = len(clauses)
    faiss = get_cached_faiss()
    nli = get_cached_nli() if run_contradiction_scoring else None

    for idx, clause_data in enumerate(clauses, start=1):
        if progress_callback is not None:
            progress_callback(idx - 1, total, f"Анализ пункта {idx}/{total}")

        clause_id = f"clause-{idx}"
        query_text = clause_data["text"]
        matches = find_matches(
            clause_id=clause_id,
            query_text=query_text,
            faiss=faiss,
            nli=nli,
            top_k=top_k,
            contradiction_model_name=contradiction_model_name,
            contradiction_threshold=contradiction_threshold,
            run_contradiction_scoring=run_contradiction_scoring,
        )
        analysis_clauses.append(
            ClauseResult(
                clause_id=clause_id,
                section=clause_data["section"],
                point_number=clause_data["point_number"],
                text=query_text,
                page_to_bbox=clause_data["page_to_bbox"],
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
            "top_k": top_k,
            "max_clauses": max_clauses,
            "contradiction_model_name": contradiction_model_name,
            "contradiction_threshold": contradiction_threshold,
            "run_contradiction_scoring": run_contradiction_scoring,
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


def find_matches(
    clause_id: str,
    query_text: str,
    faiss,
    nli: TextClassificationPipeline | None,
    top_k: int,
    contradiction_model_name: str,  # kept for response payload compatibility
    contradiction_threshold: float,
    run_contradiction_scoring: bool,
) -> list[MatchResult]:
    retrieve_results = retrieve_top_k(query_text=query_text, faiss=faiss, top_k=top_k)
    matches: list[MatchResult] = []

    for idx, item in enumerate(retrieve_results, start=1):
        meta_data = item.get("meta_data", {})
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
                match_id=f"{clause_id}-match-{idx}",
                similarity=float(item.get("similarity", 0.0)),
                norm_text=norm_text,
                hierarchy_path=list(meta_data.get("hierarchy_path", [])),
                article_number=str(meta_data.get("article_number", "")),
                part_number=str(meta_data.get("part_number", "")),
                subpart_number=str(meta_data.get("subpart_number", "")),
                contradiction_score=score,
                auto_label=auto_label,
            )
        )

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
