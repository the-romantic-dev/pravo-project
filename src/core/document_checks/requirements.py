from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import project_dir
from src.core.pdf.pdf_chunks_document import PdfChunk, normalize_text

DEFAULT_REQUIREMENTS_PATH = project_dir / "data" / "contract_requirements.json"

PLACEHOLDER_RE = re.compile(
    r"("
    r"необходимое\s+указать|"
    r"указывается|"
    r"указать|"
    r"заполняется\s+при\s+необходимости|"
    r"[\"«]\s*[\"»]|"
    r":\s*[\"«]\s*[\"»]?\s*г\.|"
    r"_{2,}|"
    r"\(\s*\)|"
    r":\s*\."
    r")",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class RequirementDefinition:
    requirement_id: str
    title: str
    description: str
    source_chunks: list[str]
    required_type: str
    patterns: list[str]
    condition_patterns: list[str]


@dataclass(slots=True)
class RequirementMatch:
    clause_id: str
    section: str
    point_number: str | None
    text: str
    matched_patterns: list[str]
    has_placeholder: bool


@dataclass(slots=True)
class RequirementCheck:
    requirement_id: str
    title: str
    description: str
    source_chunks: list[str]
    required_type: str
    status: str
    note: str
    matches: list[RequirementMatch]


def analyze_contract_requirements(
    chunks: list[PdfChunk],
    requirements_path: Path = DEFAULT_REQUIREMENTS_PATH,
) -> list[RequirementCheck]:
    requirements = load_requirements(requirements_path)
    prepared_chunks = prepare_chunks(chunks)
    full_text = normalize_text(" ".join(chunk.text for chunk in chunks))
    result = []

    for requirement in requirements:
        condition_met = is_condition_met(requirement, full_text)
        if requirement.required_type == "conditional" and not condition_met:
            result.append(
                RequirementCheck(
                    requirement_id=requirement.requirement_id,
                    title=requirement.title,
                    description=requirement.description,
                    source_chunks=requirement.source_chunks,
                    required_type=requirement.required_type,
                    status="not_applicable",
                    note="Условие применения не найдено в договоре.",
                    matches=[],
                )
            )
            continue

        matches = find_requirement_matches(requirement, prepared_chunks)
        status, note = requirement_status(requirement, matches)
        result.append(
            RequirementCheck(
                requirement_id=requirement.requirement_id,
                title=requirement.title,
                description=requirement.description,
                source_chunks=requirement.source_chunks,
                required_type=requirement.required_type,
                status=status,
                note=note,
                matches=matches,
            )
        )

    return result


@lru_cache(maxsize=1)
def load_requirements(path: Path = DEFAULT_REQUIREMENTS_PATH) -> tuple[RequirementDefinition, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requirements = []
    for item in payload:
        requirements.append(
            RequirementDefinition(
                requirement_id=str(item["id"]),
                title=str(item["title"]),
                description=str(item.get("description") or ""),
                source_chunks=[str(chunk_id) for chunk_id in item.get("source_chunks") or []],
                required_type=str(item.get("required_type") or "required"),
                patterns=[str(pattern) for pattern in item.get("patterns") or []],
                condition_patterns=[
                    str(pattern)
                    for pattern in item.get("condition_patterns") or []
                ],
            )
        )
    return tuple(requirements)


def prepare_chunks(chunks: list[PdfChunk]) -> list[dict[str, Any]]:
    prepared = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        normalized = chunk.normalized_text or normalize_text(text)
        prepared.append(
            {
                "clause_id": f"document-clause-{idx}",
                "section": chunk.heading,
                "point_number": chunk.point_number,
                "text": text,
                "normalized_text": normalized,
                "search_text": normalize_text(f"{chunk.heading} {text}"),
            }
        )
    return prepared


def is_condition_met(requirement: RequirementDefinition, full_text: str) -> bool:
    if requirement.required_type != "conditional":
        return True
    return any(regex_search(pattern, full_text) for pattern in requirement.condition_patterns)


def find_requirement_matches(
    requirement: RequirementDefinition,
    prepared_chunks: list[dict[str, Any]],
) -> list[RequirementMatch]:
    matches: list[RequirementMatch] = []
    for chunk in prepared_chunks:
        matched_patterns = [
            pattern
            for pattern in requirement.patterns
            if regex_search(pattern, chunk["search_text"])
        ]
        if not matched_patterns:
            continue

        matches.append(
            RequirementMatch(
                clause_id=str(chunk["clause_id"]),
                section=str(chunk["section"]),
                point_number=chunk["point_number"],
                text=str(chunk["text"]),
                matched_patterns=matched_patterns,
                has_placeholder=has_placeholder(str(chunk["text"])),
            )
        )

    return matches


def requirement_status(
    requirement: RequirementDefinition,
    matches: list[RequirementMatch],
) -> tuple[str, str]:
    if not matches:
        if requirement.required_type == "conditional":
            return "missing", "Есть признаки применимости условия, но подходящий пункт не найден."
        return "missing", "Подходящий пункт в договоре не найден."

    if all(match.has_placeholder for match in matches):
        return "incomplete", "Пункт найден, но выглядит незаполненным или шаблонным."

    if any(match.has_placeholder for match in matches):
        return "needs_review", "Пункт найден, но часть совпадений содержит незаполненные шаблонные поля."

    return "present", "Подходящий пункт найден."


def regex_search(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None


def has_placeholder(text: str) -> bool:
    return PLACEHOLDER_RE.search(text) is not None
