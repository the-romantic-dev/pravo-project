from __future__ import annotations

from src.ui.formatters import build_ref_for_card, format_score
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

        matches = []
        for match in clause.matches:
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

        for page_num, bbox in clause.page_to_bbox.items():
            result.setdefault(page_num, []).append(
                {
                    "bbox": bbox,
                    "clause_id": clause.clause_id,
                    "clause_text": clause.text,
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

