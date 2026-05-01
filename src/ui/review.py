from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from src.ui.formatters import build_ref, format_score, short_text
from src.ui.services import AnalysisResult, DefinitionHit, save_review

DEFAULT_OUTPUT_DIR = Path("artifacts") / "ui_annotations"


def render_results(analysis: AnalysisResult) -> None:
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
        with st.container(border=True):
            st.markdown(
                (
                    f"**Схожесть:** `{match.similarity:.3f}`  "
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

