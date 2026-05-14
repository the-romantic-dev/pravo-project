from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

from src.ui.formatters import build_ref, format_score, short_text
from src.ui.services import AnalysisResult, DefinitionHit, RequirementCheck, save_review

DEFAULT_OUTPUT_DIR = Path("artifacts") / "ui_annotations"


def render_results(analysis: AnalysisResult) -> None:
    render_document_checks(analysis.document_checks)
    st.divider()

    clause_options = [
        f"{clause.clause_id} | {clause.section} | {short_text(clause.text, limit=90)}"
        for clause in analysis.clauses
    ]
    if not clause_options:
        st.error("Не удалось извлечь пункты из PDF.")
        return

    selected_idx = st.selectbox(
        "Пункт документа",
        options=list(range(len(clause_options))),
        format_func=lambda idx: clause_options[idx],
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
        rerank = (
            f"  **Rerank:** `{match.rerank_score:.3f}`"
            if match.rerank_score is not None
            else ""
        )
        with st.container(border=True):
            st.markdown(
                (
                    f"**Схожесть:** `{match.similarity:.3f}`  "
                    f"{rerank}  "
                    f"**Авто-лейбл:** `{match.auto_label}`  "
                    f"**P(contradiction):** `{format_score(match.contradiction_score)}`"
                )
            )
            st.write(build_ref(match))
            if match.hierarchy_path:
                st.caption(" -> ".join(match.hierarchy_path))
            st.write(match.norm_text)

            current_label = labels.get(match.match_id, {}).get("manual_label", "unknown")
            label_options = {
                "unknown": "не размечено",
                "contradiction": "противоречие",
                "not_contradiction": "нет противоречия",
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


def render_document_checks(checks: list[RequirementCheck]) -> None:
    st.subheader("Обязательные условия договора")
    if not checks:
        st.info("Чеклист обязательных условий не сформирован.")
        return

    problem_statuses = {"missing", "incomplete", "needs_review"}
    active_checks = [check for check in checks if check.status != "not_applicable"]
    present_count = sum(check.status == "present" for check in active_checks)
    problem_count = sum(check.status in problem_statuses for check in active_checks)
    skipped_count = sum(check.status == "not_applicable" for check in checks)

    col_present, col_problem, col_skipped = st.columns(3)
    col_present.metric("Найдено", present_count)
    col_problem.metric("Требует внимания", problem_count)
    col_skipped.metric("Не применимо", skipped_count)

    problem_checks = [check for check in checks if check.status in problem_statuses]
    if problem_checks:
        st.warning(
            "Есть обязательные условия, которые не найдены или выглядят незаполненными."
        )
    else:
        st.success("Все применимые обязательные условия из чеклиста найдены.")

    status_order = {
        "missing": 0,
        "incomplete": 1,
        "needs_review": 2,
        "present": 3,
        "not_applicable": 4,
    }
    for check in sorted(checks, key=lambda item: (status_order.get(item.status, 9), item.title)):
        status = requirement_status_view(check.status)
        label = f"{status['label']} · {check.title}"
        with st.expander(label, expanded=check.status in problem_statuses):
            st.caption(check.description)
            st.markdown(f"**Статус:** {status['label']}")
            st.write(check.note)
            source = build_requirement_source(check)
            if source:
                st.caption(source)

            if not check.matches:
                continue

            st.markdown("**Найденные пункты договора:**")
            for match in check.matches[:5]:
                point = f", пункт {match.point_number}" if match.point_number else ""
                st.markdown(f"**{match.section}{point}**")
                st.write(match.text)
                if match.has_placeholder:
                    st.caption("Есть признаки незаполненного шаблонного поля.")

            if len(check.matches) > 5:
                st.caption(f"Показано 5 из {len(check.matches)} совпадений.")


def requirement_status_view(status: str) -> dict[str, str]:
    labels = {
        "present": "найдено",
        "missing": "не найдено",
        "incomplete": "незаполнено",
        "needs_review": "проверить",
        "not_applicable": "не применимо",
    }
    return {"label": labels.get(status, status)}


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

