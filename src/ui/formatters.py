from __future__ import annotations


def build_ref(match) -> str:
    article = match.article_number or "?"
    part = match.part_number or "?"
    subpart = match.subpart_number or "?"
    return f"Норма ТК: ст.{article}, ч.{part}, п.{subpart}"


def build_ref_for_card(match) -> str:
    article = match.article_number or "?"
    part = match.part_number or "?"
    subpart = match.subpart_number or "?"
    return f"ст.{article}, ч.{part}, п.{subpart}"


def short_text(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def format_score(score: float | None) -> str:
    if score is None:
        return "n/a"
    return f"{score:.3f}"

