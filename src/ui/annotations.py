from __future__ import annotations

from src.ui.formatters import build_ref_for_card, format_score
from src.ui.services import (
    AnalysisResult,
    find_definition_hits,
    get_cached_definition_chunk_ids,
    get_cached_definitions,
    get_cached_definitions_faiss,
)

DEFAULT_DISPLAY_DEFINITION_TOP_K = 4
DEFAULT_DISPLAY_DEFINITION_THRESHOLD = 0.72


def build_annotations_by_page(
    analysis: AnalysisResult | None,
    highlight_threshold: float,
) -> dict[int, list[dict]]:
    if analysis is None:
        return {}

    result: dict[int, list[dict]] = {}
    definition_chunk_ids = get_cached_definition_chunk_ids()
    for clause in analysis.clauses:
        if not clause.matches:
            continue

        norm_matches = [
            match
            for match in clause.matches
            if match_to_chunk_id(match) not in definition_chunk_ids
        ]
        if not norm_matches:
            continue

        best_match = max(norm_matches, key=lambda match: match.similarity)
        if best_match.similarity < highlight_threshold:
            continue

        matches = []
        for match in norm_matches:
            status = map_status(match.auto_label)
            matches.append(
                {
                    "tk_ref": build_ref_for_card(match),
                    "tk_text": (match.norm_text or "").strip(),
                    "similarity": f"{match.similarity:.3f}",
                    "contradiction_score": format_score(match.contradiction_score),
                    "status_text": status["text"],
                    "status_color": status["color"],
                }
            )

        definitions = []
        for definition in get_clause_definitions(clause):
            definitions.append(
                {
                    "term": definition.term,
                    "text": definition.text,
                    "source": build_definition_source(definition.source_articles),
                    "similarity": (
                        f"{definition.similarity:.3f}"
                        if definition.similarity is not None
                        else "n/a"
                    ),
                    "match_type": definition.match_type,
                }
            )

        for page_num, bbox in clause.page_to_bbox.items():
            result.setdefault(page_num, []).append(
                {
                    "bbox": bbox,
                    "clause_id": clause.clause_id,
                    "clause_text": clause.text,
                    "definitions": definitions,
                    "matches": matches,
                }
            )

    for page_num in result:
        result[page_num].sort(key=lambda item: (item["bbox"][1] + item["bbox"][3]) / 2)
    return result


def map_status(auto_label: str) -> dict[str, str]:
    if auto_label == "contradiction":
        return {"text": "Противоречит", "color": "#d94841"}
    if auto_label == "not_contradiction":
        return {"text": "Не противоречит", "color": "#2f9e44"}
    return {"text": "Не оценено", "color": "#868e96"}


def get_clause_definitions(clause):
    definitions = getattr(clause, "definitions", None)
    if definitions:
        return definitions

    try:
        return find_definition_hits(
            query_text=clause.text,
            definitions_faiss=get_cached_definitions_faiss(),
            definitions=get_cached_definitions(),
            top_k=DEFAULT_DISPLAY_DEFINITION_TOP_K,
            similarity_threshold=DEFAULT_DISPLAY_DEFINITION_THRESHOLD,
        )
    except Exception:
        return []


def match_to_chunk_id(match) -> str:
    return f"art:{match.article_number}:part:{match.part_number}:sub:{match.subpart_number}"


def build_definition_source(source_articles: list[str]) -> str:
    if not source_articles:
        return ""
    return "Источник: " + ", ".join(f"ст. {article}" for article in source_articles)

