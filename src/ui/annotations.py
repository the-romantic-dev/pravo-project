from __future__ import annotations

from src.ui.formatters import build_ref_for_card
from src.ui.services import AnalysisResult


def build_annotations_by_page(
    analysis: AnalysisResult | None,
    highlight_threshold: float,
) -> dict[int, list[dict]]:
    if analysis is None:
        return {}

    result: dict[int, list[dict]] = {}
    for clause in analysis.clauses:
        if not clause.matches:
            continue

        best_match = max(clause.matches, key=lambda match: match.similarity)
        if best_match.similarity < highlight_threshold:
            continue

        tk_ref = build_ref_for_card(best_match)
        status = map_status(best_match.auto_label)
        tk_text = (best_match.norm_text or "").strip()

        for page_num, bbox in clause.page_to_bbox.items():
            result.setdefault(page_num, []).append(
                {
                    "bbox": bbox,
                    "tk_ref": tk_ref,
                    "tk_text": tk_text,
                    "status_text": status["text"],
                    "status_color": status["color"],
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

