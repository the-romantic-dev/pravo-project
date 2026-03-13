from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(slots=True)
class SidebarControls:
    run_scoring: bool
    contradiction_threshold: float
    highlight_threshold: float


def render_sidebar_controls() -> SidebarControls:
    with st.sidebar:
        st.header("Параметры анализа")
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
        run_scoring=run_scoring,
        contradiction_threshold=float(contradiction_threshold),
        highlight_threshold=float(highlight_threshold),
    )
