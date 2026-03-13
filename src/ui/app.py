from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if PROJECT_ROOT.as_posix() not in [Path(p).as_posix() for p in sys.path if p]:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.ui.controls import render_sidebar_controls
from src.ui.pdf_viewer import show_pdf
from src.ui.review import render_results, save_current_review
from src.ui.services import analyze_pdf


def main() -> None:
    st.set_page_config(
        page_title="TKRF: проверка противоречий",
        layout="wide",
    )
    st.title("Проверка противоречий документа с нормами ТК РФ")

    controls = render_sidebar_controls()
    uploaded_pdf = st.file_uploader(
        "Загрузите PDF документ для проверки",
        type=["pdf"],
    )
    if uploaded_pdf is None:
        st.info("Загрузите PDF, чтобы запустить анализ и просмотреть совпадения.")
        return

    pdf_bytes = uploaded_pdf.getvalue()
    temp_pdf = Path("artifacts") / "tmp" / uploaded_pdf.name
    temp_pdf.parent.mkdir(parents=True, exist_ok=True)
    temp_pdf.write_bytes(pdf_bytes)

    run_analysis = st.button("Запустить анализ", type="primary")
    if run_analysis:
        progress = st.progress(0, text="Подготовка анализа...")

        def on_progress(done: int, total: int, message: str) -> None:
            ratio = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
            progress.progress(ratio, text=message)

        with st.spinner("Выполняю поиск совпадений и оценку противоречий..."):
            analysis = analyze_pdf(
                pdf_path=temp_pdf,
                top_k=1,
                max_clauses=None,
                contradiction_threshold=controls.contradiction_threshold,
                run_contradiction_scoring=controls.run_scoring,
                progress_callback=on_progress,
            )
        progress.progress(1.0, text="Анализ завершен")
        st.session_state["analysis"] = analysis
        st.session_state["labels"] = {}

    analysis = st.session_state.get("analysis")
    with st.expander("Просмотр PDF", expanded=True):
        show_pdf(
            pdf_bytes=pdf_bytes,
            file_name=uploaded_pdf.name,
            analysis=analysis,
            highlight_threshold=controls.highlight_threshold,
        )

    if analysis:
        render_results(analysis)
        if st.button("Сохранить разметку", type="secondary"):
            saved_path = save_current_review(analysis)
            st.success(f"Разметка сохранена: {saved_path.as_posix()}")
    else:
        st.warning("Нажмите «Запустить анализ», чтобы увидеть результаты.")


if __name__ == "__main__":
    main()
