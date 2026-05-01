from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(slots=True)
class SidebarControls:
    top_k: int
    definition_top_k: int
    definition_similarity_threshold: float
    run_scoring: bool
    contradiction_threshold: float
    highlight_threshold: float


def render_sidebar_controls() -> SidebarControls:
    with st.sidebar:
        st.header("Параметры анализа")
        top_k = st.slider(
            "Количество норм ТК для пункта",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )
        definition_top_k = st.slider(
            "Количество определений для пункта",
            min_value=0,
            max_value=10,
            value=4,
            step=1,
        )
        definition_similarity_threshold = st.slider(
            "Порог похожести определений",
            min_value=-1.0,
            max_value=1.0,
            value=0.72,
            step=0.01,
            disabled=definition_top_k == 0,
        )
        run_scoring = st.checkbox(
            "Считать вероятность противоречия автоматически",
            value=True,
        )
        contradiction_threshold = st.slider(
            "Порог противоречия",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            disabled=not run_scoring,
        )
        highlight_threshold = st.slider(
            "Порог подсветки bbox (по похожести)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.01,
        )

    return SidebarControls(
        top_k=int(top_k),
        definition_top_k=int(definition_top_k),
        definition_similarity_threshold=float(definition_similarity_threshold),
        run_scoring=run_scoring,
        contradiction_threshold=float(contradiction_threshold),
        highlight_threshold=float(highlight_threshold),
    )
