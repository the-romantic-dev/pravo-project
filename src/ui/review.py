from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.formatters import build_ref, format_score, short_text
from src.ui.services import AnalysisResult, save_review

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


def save_current_review(analysis: AnalysisResult) -> Path:
    output_path = DEFAULT_OUTPUT_DIR / f"{analysis.analysis_id}_{analysis.pdf_name}.json"
    labels = st.session_state.get("labels", {})
    return save_review(output_path=output_path, analysis=analysis, labels=labels)

