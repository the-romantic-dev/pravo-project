from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st

from src.config import rag_chunks_path
from src.ui.formatters import build_ref, format_score, short_text
from src.ui.services import AnalysisResult, DefinitionHit, RequirementCheck, save_review

DEFAULT_OUTPUT_DIR = Path("artifacts") / "ui_annotations"
CLAUSE_SELECT_KEY = "selected_clause_index"


def render_results(analysis: AnalysisResult) -> None:
    clause_options = [
        f"{clause.clause_id} | {clause.section} | {short_text(clause.text, limit=90)}"
        for clause in analysis.clauses
    ]
    if not clause_options:
        st.error("Не удалось извлечь пункты из PDF.")
        return

    valid_indexes = set(range(len(clause_options)))
    if st.session_state.get(CLAUSE_SELECT_KEY) not in valid_indexes:
        st.session_state[CLAUSE_SELECT_KEY] = 0

    selected_idx = st.selectbox(
        "Пункт документа",
        options=list(range(len(clause_options))),
        format_func=lambda idx: clause_options[idx],
        key=CLAUSE_SELECT_KEY,
    )
    clause = analysis.clauses[selected_idx]

    st.subheader("Текст пункта")
    st.write(clause.text)
    st.caption(f"Раздел: {clause.section}")
    render_definition_chips(clause.definitions)

    st.subheader("Найденные совпадения")
    if not clause.matches:
        st.info("Совпадения не найдены.")
        return

    labels = st.session_state.setdefault("labels", {})
    for match in clause.matches:
        with st.container(border=True):
            status = nli_status_view(match.auto_label)
            st.markdown(
                (
                    '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;'
                    'margin:0 0 8px 0;font-size:13px;">'
                    f'<span><strong>Схожесть:</strong> <code>{match.similarity:.3f}</code></span>'
                    + (
                        f'<span><strong>Rerank:</strong> <code>{match.rerank_score:.3f}</code></span>'
                        if match.rerank_score is not None
                        else ""
                    )
                    + (
                        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
                        f'background:{status["background"]};color:{status["color"]};'
                        f'font-weight:700;">{html.escape(status["label"])}</span>'
                    )
                    + f'<span><strong>E:</strong> <code>{format_score(match.entailment_score)}</code></span>'
                    + f'<span><strong>N:</strong> <code>{format_score(match.neutral_score)}</code></span>'
                    + f'<span><strong>C:</strong> <code>{format_score(match.contradiction_score)}</code></span>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
            st.write(build_ref(match))
            if match.hierarchy_path:
                st.caption(" -> ".join(match.hierarchy_path))
            st.write(match.norm_text)

            current_label = labels.get(match.match_id, {}).get("manual_label", "unknown")
            if current_label == "not_contradiction":
                current_label = "entailment"
            label_options = {
                "unknown": "не размечено",
                "entailment": "соответствие",
                "neutral": "нейтрально",
                "contradiction": "противоречие",
            }
            selected_label = st.radio(
                "Ручная пометка",
                options=list(label_options.keys()),
                format_func=lambda key: label_options[key],
                index=list(label_options.keys()).index(current_label),
                key=f"label-{match.match_id}",
                horizontal=True,
            )
            comment = st.text_input(
                "Комментарий",
                value=labels.get(match.match_id, {}).get("comment", ""),
                key=f"comment-{match.match_id}",
            )
            labels[match.match_id] = {
                "manual_label": selected_label,
                "comment": comment,
            }


def nli_status_view(auto_label: str) -> dict[str, str]:
    statuses = {
        "entailment": {
            "label": "entailment",
            "color": "#2f9e44",
            "background": "#ebfbee",
        },
        "neutral": {
            "label": "neutral",
            "color": "#868e96",
            "background": "#f1f3f5",
        },
        "contradiction": {
            "label": "contradiction",
            "color": "#d94841",
            "background": "#fff5f5",
        },
        # Backward compatibility for analyses saved before schema v9.
        "not_contradiction": {
            "label": "entailment",
            "color": "#2f9e44",
            "background": "#ebfbee",
        },
    }
    return statuses.get(
        auto_label,
        {"label": "not_scored", "color": "#868e96", "background": "#f1f3f5"},
    )


def render_document_checks(analysis: AnalysisResult) -> None:
    checks = analysis.document_checks
    if not checks:
        return

    problem_statuses = {"missing"}
    active_checks = [check for check in checks if check.status != "not_applicable"]
    present_count = sum(check.status == "present" for check in active_checks)
    problem_count = sum(check.status in problem_statuses for check in active_checks)
    skipped_count = sum(check.status == "not_applicable" for check in checks)

    status_order = {
        "present": 0,
        "missing": 1,
        "not_applicable": 3,
    }
    compact_checks = sorted(
        checks,
        key=lambda item: (status_order.get(item.status, 9), item.title),
    )

    chips = []
    for check in compact_checks:
        status = requirement_status_view(check.status)
        chips.append(
            (
                f'<span class="doc-check-item doc-check-item--{check.status}" tabindex="0">'
                f'<span class="doc-check-item__label">'
                f'{html.escape(status["mark"])} {html.escape(check.title)}'
                "</span>"
                f'<span class="doc-check-tooltip">'
                f'{build_requirement_tooltip_html(check)}'
                "</span>"
                "</span>"
            )
        )

    st.markdown(
        (
            """
        <style>
        .doc-check-panel {
            margin: 0 0 12px 0;
        }
        .doc-check-panel__header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 16px;
            margin: 0 0 8px 0;
        }
        .doc-check-panel__title {
            font-weight: 700;
            color: inherit;
        }
        .doc-check-panel__summary {
            color: #7d8593;
            font-size: 13px;
            white-space: nowrap;
        }
        .doc-check-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            padding: 2px 0 12px 0;
        }
        .doc-check-item {
            position: relative;
            min-width: 0;
            min-height: 34px;
            box-sizing: border-box;
            padding: 7px 11px;
            border: 1px solid #d7dde6;
            border-radius: 8px;
            background: #f8fafc;
            color: #344054;
            font-size: 13px;
            line-height: 1.25;
            white-space: nowrap;
            cursor: default;
        }
        .doc-check-item__label {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .doc-check-item--present {
            border-color: #a7d8b5;
            background: #f1fbf4;
            color: #176b32;
        }
        .doc-check-item--missing {
            border-color: #f1b8b3;
            background: #fff5f5;
            color: #9f1f18;
        }
        .doc-check-item--not_applicable {
            color: #788392;
            background: #f5f6f8;
        }
        @media (max-width: 760px) {
            .doc-check-row {
                grid-template-columns: 1fr;
            }
        }
        .doc-check-tooltip {
            display: none;
            position: fixed;
            left: 48px;
            right: 48px;
            top: 138px;
            z-index: 2000;
            max-height: min(58vh, 560px);
            overflow: auto;
            box-sizing: border-box;
            padding: 14px 16px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            background: #ffffff;
            color: #172033;
            box-shadow: 0 18px 42px rgba(15, 23, 42, .25);
            white-space: normal;
            font-size: 13px;
            line-height: 1.45;
        }
        .doc-check-item:hover .doc-check-tooltip,
        .doc-check-item:focus .doc-check-tooltip {
            display: block;
        }
        .doc-check-tooltip__title {
            display: block;
            margin-bottom: 6px;
            font-size: 14px;
            font-weight: 800;
            color: #0f172a;
        }
        .doc-check-tooltip__note {
            display: block;
            margin-bottom: 10px;
            color: #475569;
        }
        .doc-check-tooltip__source {
            display: block;
            padding-top: 10px;
            margin-top: 10px;
            border-top: 1px solid #e2e8f0;
        }
        .doc-check-tooltip__path {
            display: block;
            margin-bottom: 6px;
            color: #64748b;
            font-size: 12px;
        }
        .doc-check-tooltip__text {
            display: block;
            color: #172033;
        }
        </style>
        """
            '<div class="doc-check-panel">'
            '<div class="doc-check-panel__header">'
            '<span class="doc-check-panel__title">Проверки уровня документа</span>'
            f'<span class="doc-check-panel__summary">'
            f'✅ {present_count} · ⚠️ {problem_count} · ➖ {skipped_count}'
            "</span>"
            "</div>"
            '<div class="doc-check-row">'
            + "".join(chips)
            + "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def requirement_status_view(status: str) -> dict[str, str]:
    labels = {
        "present": "найдено",
        "missing": "не найдено",
        "not_applicable": "не применимо",
    }
    marks = {
        "present": "✅",
        "missing": "❌",
        "not_applicable": "➖",
    }
    return {"label": labels.get(status, status), "mark": marks.get(status, status)}


def build_requirement_tooltip_html(check: RequirementCheck) -> str:
    status = requirement_status_view(check.status)
    sources = source_chunks_for_requirement(check)
    source_html_parts = []

    for chunk in sources[:8]:
        hierarchy = " -> ".join(str(item) for item in chunk.get("hierarchy_path", []) if item)
        text = str(chunk.get("original_text") or chunk.get("normalized_text") or "").strip()
        source_html_parts.append(
            (
                '<span class="doc-check-tooltip__source">'
                f'<span class="doc-check-tooltip__path">{html.escape(hierarchy)}</span>'
                f'<span class="doc-check-tooltip__text">{html.escape(text)}</span>'
                "</span>"
            )
        )

    if len(sources) > 8:
        source_html_parts.append(
            (
                '<span class="doc-check-tooltip__source">'
                f'<span class="doc-check-tooltip__path">Еще {len(sources) - 8} норм не показано</span>'
                "</span>"
            )
        )

    if not source_html_parts:
        source_html_parts.append(
            (
                '<span class="doc-check-tooltip__source">'
                f'<span class="doc-check-tooltip__path">{html.escape(build_requirement_source(check))}</span>'
                "</span>"
            )
        )

    return (
        f'<span class="doc-check-tooltip__title">'
        f'{html.escape(status["mark"])} {html.escape(check.title)}'
        "</span>"
        f'<span class="doc-check-tooltip__note">{html.escape(check.note)}</span>'
        + "".join(source_html_parts)
    )


def source_chunks_for_requirement(check: RequirementCheck) -> list[dict]:
    by_id = load_tk_chunks_by_id()
    by_article = load_tk_chunks_by_article()
    chunks = []
    seen: set[str] = set()

    for source_chunk in check.source_chunks:
        if source_chunk.endswith(":*"):
            article = source_chunk.removeprefix("art:").removesuffix(":*")
            for chunk in by_article.get(article, []):
                chunk_id = str(chunk.get("chunk_id") or "")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append(chunk)
            continue

        chunk = by_id.get(source_chunk)
        if chunk is None or source_chunk in seen:
            continue
        seen.add(source_chunk)
        chunks.append(chunk)

    return chunks


@lru_cache(maxsize=1)
def load_tk_chunks_by_id() -> dict[str, dict]:
    chunks = load_tk_chunks()
    return {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}


@lru_cache(maxsize=1)
def load_tk_chunks_by_article() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for chunk in load_tk_chunks():
        article = str(chunk.get("article_number") or "")
        if article:
            result.setdefault(article, []).append(chunk)
    return result


@lru_cache(maxsize=1)
def load_tk_chunks() -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for line in rag_chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_requirement_source(check: RequirementCheck) -> str:
    refs = [format_source_chunk(chunk_id) for chunk_id in check.source_chunks]
    refs = [ref for ref in refs if ref]
    if not refs:
        return ""
    return "Источник: " + ", ".join(refs)


def format_source_chunk(chunk_id: str) -> str:
    if chunk_id.endswith(":*"):
        article = chunk_id.removeprefix("art:").removesuffix(":*")
        return f"ст. {article}"

    match = re.match(r"art:([^:]+):part:([^:]+):sub:([^:]+)", chunk_id)
    if not match:
        return chunk_id
    article, part, subpart = match.groups()
    return f"ст. {article}, ч. {part}, п. {subpart}"


def render_definition_chips(definitions: list[DefinitionHit]) -> None:
    if not definitions:
        return

    chips = []
    for definition in definitions:
        source = build_definition_source(definition)
        score = (
            f"Похожесть: {definition.similarity:.3f}"
            if definition.similarity is not None
            else ""
        )
        details = "\n".join(
            part
            for part in [
                definition.text,
                source,
                score,
            ]
            if part
        )
        chips.append(
            (
                '<span class="definition-chip" tabindex="0">'
                f'{html.escape(definition.term)}'
                '<span class="definition-tooltip">'
                f'<span class="definition-tooltip__title">{html.escape(definition.term)}</span>'
                f'<span class="definition-tooltip__text">{html.escape(details)}</span>'
                "</span>"
                "</span>"
            )
        )

    st.markdown(
        (
            "<style>"
            ".definition-panel{margin:6px 0 18px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center;}"
            ".definition-panel__label{font-size:13px;color:#5c6773;margin-right:2px;}"
            ".definition-chip{position:relative;display:inline-flex;align-items:center;max-width:100%;"
            "padding:4px 10px;border:1px solid #b9d8cf;border-radius:999px;background:#eef8f5;"
            "color:#0f766e;font-size:13px;font-weight:600;line-height:1.3;cursor:default;}"
            ".definition-chip:focus{outline:2px solid #14b8a6;outline-offset:2px;}"
            ".definition-tooltip{display:none;position:absolute;left:0;top:calc(100% + 8px);z-index:1000;"
            "width:min(560px,78vw);max-height:280px;overflow:auto;padding:12px 14px;border:1px solid #c9d6df;"
            "border-radius:8px;background:#ffffff;color:#111827;box-shadow:0 12px 30px rgba(15,23,42,.18);"
            "font-weight:400;line-height:1.45;white-space:pre-wrap;}"
            ".definition-tooltip__title{display:block;margin-bottom:6px;font-weight:700;color:#0f172a;}"
            ".definition-tooltip__text{display:block;color:#334155;}"
            ".definition-chip:hover .definition-tooltip,.definition-chip:focus .definition-tooltip{display:block;}"
            "</style>"
            '<div class="definition-panel">'
            '<span class="definition-panel__label">Найденные понятия:</span>'
            + "".join(chips)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )


def build_definition_source(definition: DefinitionHit) -> str:
    if definition.source_articles:
        articles = ", ".join(f"ст. {article}" for article in definition.source_articles)
        return f"Источник: {articles}"
    if definition.chunks:
        return "Источник: " + ", ".join(definition.chunks)
    return ""


def save_current_review(analysis: AnalysisResult) -> Path:
    output_path = DEFAULT_OUTPUT_DIR / f"{analysis.analysis_id}_{analysis.pdf_name}.json"
    labels = st.session_state.get("labels", {})
    return save_review(output_path=output_path, analysis=analysis, labels=labels)

