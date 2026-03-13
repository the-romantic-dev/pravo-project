from __future__ import annotations

import base64
import html
import io

import streamlit as st

from src.ui.annotations import build_annotations_by_page
from src.ui.services import AnalysisResult


def show_pdf(
    pdf_bytes: bytes,
    file_name: str,
    analysis: AnalysisResult | None = None,
    highlight_threshold: float = 0.8,
) -> None:
    st.download_button(
        "Скачать PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )
    try:
        import pypdfium2 as pdfium
    except Exception:
        st.warning("Не удалось загрузить модуль рендера PDF. Используйте скачивание файла.")
        return

    try:
        document = pdfium.PdfDocument(pdf_bytes)
    except Exception as error:
        st.error(f"Не удалось открыть PDF: {error}")
        return

    page_count = len(document)
    if page_count == 0:
        st.info("PDF не содержит страниц.")
        return

    render_scale = 3.0
    start_idx = 0
    end_idx = page_count
    annotations_by_page = build_annotations_by_page(analysis, highlight_threshold)

    pages_payload: list[dict] = []
    try:
        for page_index in range(start_idx, end_idx):
            page = document[page_index]
            bitmap = page.render(scale=float(render_scale))
            image = bitmap.to_pil()
            pages_payload.append(
                {
                    "page_number": page_index + 1,
                    "display_width": int(image.width / float(render_scale)),
                    "display_height": int(image.height / float(render_scale)),
                    "image_base64": pil_to_base64(image),
                    "annotations": annotations_by_page.get(page_index, []),
                }
            )
            bitmap.close()
            page.close()
    finally:
        document.close()

    render_synced_pages_viewer(pages_payload=pages_payload)


def render_synced_pages_viewer(pages_payload: list[dict]) -> None:
    if not pages_payload:
        st.info("Нет страниц для отображения.")
        return

    page_gap = 22
    right_panel_width = 430
    pane_gap = 14
    doc_width = max(p["display_width"] for p in pages_payload)
    wrapper_id = "sync-wrap-global"
    left_pane_id = "left-pane-global"
    right_pane_id = "right-pane-global"
    right_cards_id = "right-cards-global"
    link_svg_id = "link-svg-global"

    pages_html_parts: list[str] = []
    annotations: list[dict] = []
    y_cursor = 0

    for page in pages_payload:
        page_top = y_cursor
        page_w = page["display_width"]
        page_h = page["display_height"]
        x_offset = max(0, int((doc_width - page_w) / 2))

        pages_html_parts.append(
            (
                f'<div style="position:absolute;left:{x_offset}px;top:{page_top}px;'
                f'width:{page_w}px;height:{page_h}px;">'
                f'<img src="data:image/png;base64,{page["image_base64"]}" '
                f'style="width:{page_w}px;height:{page_h}px;display:block;background:#fff;border-radius:8px;" />'
                f'<div style="position:absolute;left:8px;bottom:8px;padding:2px 8px;border-radius:999px;'
                'background:rgba(0,0,0,.45);color:#fff;font-size:11px;">'
                f'Страница {page["page_number"]}'
                "</div></div>"
            )
        )

        for idx, item in enumerate(page["annotations"]):
            x0, y0, x1, y1 = item["bbox"]
            ann_id = f"ann-{page['page_number']}-{idx}"
            annotations.append(
                {
                    "id": ann_id,
                    "x0": x_offset + x0,
                    "y0": page_top + y0,
                    "x1": x_offset + x1,
                    "y1": page_top + y1,
                    "y_center_px": page_top + (y0 + y1) * 0.5,
                    "tk_ref": item["tk_ref"],
                    "tk_text": item["tk_text"],
                    "status_text": item["status_text"],
                    "status_color": item["status_color"],
                }
            )

        y_cursor += page_h + page_gap

    doc_height = max(0, y_cursor - page_gap)
    svg_overlay = build_overlay_svg(
        image_width=doc_width,
        image_height=doc_height,
        annotations=annotations,
    )
    cards_html = build_cards_html(
        annotations=annotations,
        card_width=right_panel_width - 20,
    )
    sync_script = build_sync_script(
        wrapper_id=wrapper_id,
        left_pane_id=left_pane_id,
        right_pane_id=right_pane_id,
        right_cards_id=right_cards_id,
        link_svg_id=link_svg_id,
    )

    page_html = (
        f'<div id="{wrapper_id}" style="width:100%;height:min(84vh,920px);display:flex;gap:{pane_gap}px;'
        'border:1px solid #1f2d3d;border-radius:10px;background:#0b1220;padding:8px;box-sizing:border-box;position:relative;">'
        f'<svg id="{link_svg_id}" style="position:absolute;left:0;top:0;width:100%;height:100%;'
        'pointer-events:none;z-index:6;"></svg>'
        f'<div id="{left_pane_id}" style="flex:0 0 62%;overflow:auto;position:relative;border-radius:8px;background:#f8f9fa;">'
        f'<div style="position:relative;width:{doc_width}px;height:{doc_height}px;">'
        + "".join(pages_html_parts)
        + f"{svg_overlay}</div></div>"
        f'<div id="{right_pane_id}" style="flex:1;overflow:hidden;position:relative;border-radius:8px;'
        'background:#0f1726;padding:8px 8px 12px 8px;">'
        '<div style="color:#e9ecef;font-size:13px;font-weight:600;margin:2px 4px 10px 4px;">'
        "Соответствующие нормы ТК"
        "</div>"
        f'<div id="{right_cards_id}" style="position:relative;width:100%;height:calc(100% - 28px);overflow:hidden;">'
        f"{cards_html}</div></div></div>{sync_script}"
    )
    st.components.v1.html(page_html, height=min(980, max(500, doc_height + 88)), scrolling=False)


def pil_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_overlay_svg(
    image_width: int,
    image_height: int,
    annotations: list[dict],
) -> str:
    boxes: list[str] = []
    for item in annotations:
        boxes.append(
            f'<rect id="box-{item["id"]}" data-ann-id="{item["id"]}" data-center-y="{item["y_center_px"]:.2f}" '
            f'x="{item["x0"]:.1f}" y="{item["y0"]:.1f}" '
            f'width="{max(1.0, item["x1"] - item["x0"]):.1f}" '
            f'height="{max(1.0, item["y1"] - item["y0"]):.1f}" '
            'fill="rgba(22,160,133,0.07)" stroke="#16a085" stroke-width="2.5" rx="2" />'
        )

    return (
        f'<svg width="{image_width}" height="{image_height}" '
        'style="position:absolute;left:0;top:0;pointer-events:auto;">'
        + "".join(boxes)
        + "</svg>"
    )


def build_cards_html(
    annotations: list[dict],
    card_width: int,
) -> str:
    parts: list[str] = []
    for item in annotations:
        parts.append(
            (
                f'<div id="card-{item["id"]}" data-ann-id="{item["id"]}" '
                f'data-doc-y="{item["y_center_px"]:.2f}" '
                f'style="position:absolute;left:0;top:0;width:{card_width}px;display:none;'
                'border:1px solid #2a3b52;border-radius:10px;padding:10px 12px;'
                'background:#111b2e;color:#e9ecef;font-size:12px;line-height:1.25;box-shadow:0 2px 8px rgba(0,0,0,.18);'
                'transition:all .15s ease;box-sizing:border-box;">'
                f'<div style="font-weight:600;margin-bottom:4px;">{html.escape(item["tk_ref"])}</div>'
                f'<div style="color:#cfd8e3;margin-bottom:8px;">{html.escape(item["tk_text"])}</div>'
                f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
                f'background:{item["status_color"]}22;color:{item["status_color"]};font-weight:600;">'
                f'{html.escape(item["status_text"])}</span>'
                "</div>"
            )
        )
    return "".join(parts)


def build_sync_script(
    wrapper_id: str,
    left_pane_id: str,
    right_pane_id: str,
    right_cards_id: str,
    link_svg_id: str,
) -> str:
    return f"""
<style>
#{wrapper_id} .active-card {{
  border-color:#5c7cfa !important;
  box-shadow:0 0 0 1px #5c7cfa44, 0 4px 14px rgba(0,0,0,.28) !important;
  transform:translateY(-1px);
}}
#{wrapper_id} .active-box {{
  stroke:#5c7cfa !important;
  stroke-width:3.2 !important;
  fill:rgba(92,124,250,0.12) !important;
}}
</style>
<script>
(function() {{
  const left = document.getElementById("{left_pane_id}");
  const right = document.getElementById("{right_pane_id}");
  const rightCards = document.getElementById("{right_cards_id}");
  const wrapper = document.getElementById("{wrapper_id}");
  const linkSvg = document.getElementById("{link_svg_id}");
  if (!left || !right || !rightCards || !wrapper || !linkSvg) return;

  const cards = Array.from(right.querySelectorAll("[id^='card-'][data-ann-id]"));
  const boxes = Array.from(left.querySelectorAll("[id^='box-'][data-ann-id]"));
  if (!cards.length || !boxes.length) return;

  const cardById = new Map(cards.map((c) => [c.dataset.annId, c]));
  const boxById = new Map(boxes.map((b) => [b.dataset.annId, b]));
  const cardsSorted = [...cards].sort(
    (a, b) => parseFloat(a.dataset.docY || "0") - parseFloat(b.dataset.docY || "0")
  );
  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const docY = (card) => parseFloat(card.dataset.docY || "0");

  let isProgrammatic = false;
  let framePending = false;
  let activeId = cardsSorted[0].dataset.annId;
  let wheelAccum = 0;
  let wheelLocked = false;

  function viewport() {{
    const top = left.scrollTop;
    const bottom = top + left.clientHeight;
    return {{ top, bottom, center: (top + bottom) / 2 }};
  }}

  function pickVisibleCards() {{
    const idx = Math.max(
      0,
      cardsSorted.findIndex((card) => card.dataset.annId === activeId)
    );
    const prev = idx > 0 ? cardsSorted[idx - 1] : null;
    const curr = cardsSorted[idx];
    const next = idx + 1 < cardsSorted.length ? cardsSorted[idx + 1] : null;
    return [prev, curr, next].filter(Boolean);
  }}

  function renderConnections(visibleCards) {{
    const wrapperRect = wrapper.getBoundingClientRect();
    const leftRect = left.getBoundingClientRect();
    const rightCardsRect = rightCards.getBoundingClientRect();
    const leftTopInWrap = leftRect.top - wrapperRect.top;
    const leftBottomInWrap = leftRect.bottom - wrapperRect.top;
    const leftOffsetX = leftRect.left - wrapperRect.left;
    const rightCardsOffsetX = rightCardsRect.left - wrapperRect.left;
    const rightCardsOffsetY = rightCardsRect.top - wrapperRect.top;
    let paths = "";
    for (const card of visibleCards) {{
      const id = card.dataset.annId;
      const box = boxById.get(id);
      if (!box) continue;

      const boxX = parseFloat(box.getAttribute("x") || "0");
      const boxW = parseFloat(box.getAttribute("width") || "0");
      const boxCenterY = parseFloat(box.dataset.centerY || "0");
      const cardCenterY = parseFloat(card.dataset.cardY || "0");

      const x1 = leftOffsetX + boxX + boxW;
      const y1 = leftTopInWrap + boxCenterY - left.scrollTop;
      const x2 = rightCardsOffsetX;
      const y2 = rightCardsOffsetY + cardCenterY;

      if (y1 < leftTopInWrap || y1 > leftBottomInWrap) {{
        continue;
      }}

      const dx = Math.max(40, (x2 - x1) * 0.35);
      paths += `<path d="M ${{x1.toFixed(1)}} ${{y1.toFixed(1)}} C ${{(x1 + dx).toFixed(1)}} ${{y1.toFixed(1)}}, ${{(x2 - dx).toFixed(1)}} ${{y2.toFixed(1)}}, ${{x2.toFixed(1)}} ${{y2.toFixed(1)}}" stroke="#74c0fc" stroke-width="1.8" fill="none" stroke-linecap="round" opacity="0.88" />`;
    }}
    linkSvg.innerHTML = paths;
  }}

  function layoutCards() {{
    const margin = 6;
    const gap = 8;
    const paneH = Math.max(120, rightCards.clientHeight - 8);
    const visibleCards = pickVisibleCards();

    cards.forEach((card) => {{
      if (visibleCards.includes(card)) {{
        card.style.display = "block";
      }} else {{
        card.style.display = "none";
        card.classList.remove("active-card");
      }}
    }});

    if (!visibleCards.length) {{
      renderConnections([]);
      return;
    }}

    const activeCard = visibleCards.find((card) => card.dataset.annId === activeId) || visibleCards[0];
    const activeH = activeCard.offsetHeight || 90;
    const activeDocY = docY(activeCard);
    const activeDesiredTop = activeDocY - left.scrollTop - activeH / 2;
    const activeTop = clamp(activeDesiredTop, margin, Math.max(margin, paneH - activeH - margin));

    const byYAsc = (a, b) => docY(a) - docY(b);
    const prevCards = visibleCards
      .filter((card) => docY(card) < activeDocY)
      .sort((a, b) => docY(b) - docY(a));
    const nextCards = visibleCards
      .filter((card) => docY(card) > activeDocY)
      .sort(byYAsc);

    const positions = new Map();
    positions.set(activeCard, activeTop);

    let upCursor = activeTop - gap;
    for (const card of prevCards) {{
      const h = card.offsetHeight || 90;
      const top = Math.max(margin, upCursor - h);
      positions.set(card, top);
      upCursor = top - gap;
    }}

    let downCursor = activeTop + activeH + gap;
    for (const card of nextCards) {{
      const h = card.offsetHeight || 90;
      const top = Math.min(Math.max(margin, paneH - h - margin), downCursor);
      positions.set(card, top);
      downCursor = top + h + gap;
    }}

    for (const card of visibleCards) {{
      const h = card.offsetHeight || 90;
      const top = positions.has(card) ? positions.get(card) : margin;
      card.style.transform = `translateY(${{Math.round(top)}}px)`;
      card.dataset.cardY = String(top + h / 2);
    }}
    renderConnections(visibleCards);
  }}

  function activate(id, alignToCenter) {{
    activeId = id;
    cards.forEach((el) => el.classList.toggle("active-card", el.dataset.annId === id));
    boxes.forEach((el) => el.classList.toggle("active-box", el.dataset.annId === id));

    const card = cardById.get(id);
    if (!card) return;
    if (!alignToCenter) return;

    const target = docY(card) - left.clientHeight / 2;
    isProgrammatic = true;
    left.scrollTop = clamp(target, 0, Math.max(0, left.scrollHeight - left.clientHeight));
    layoutCards();
    setTimeout(() => {{ isProgrammatic = false; }}, 60);
  }}

  function nearestByCenter() {{
    const vp = viewport();
    const edgePx = Math.max(12, left.clientHeight * 0.05);
    if (vp.top <= edgePx) {{
      return cardsSorted[0].dataset.annId;
    }}
    if (left.scrollHeight - vp.bottom <= edgePx) {{
      return cardsSorted[cardsSorted.length - 1].dataset.annId;
    }}

    const source = cardsSorted;
    let bestId = source[0].dataset.annId;
    let bestDist = Number.POSITIVE_INFINITY;
    let activeDist = Number.POSITIVE_INFINITY;
    source.forEach((card) => {{
      const dist = Math.abs(docY(card) - vp.center);
      if (card.dataset.annId === activeId) {{
        activeDist = dist;
      }}
      if (dist < bestDist) {{
        bestDist = dist;
        bestId = card.dataset.annId;
      }}
    }});

    if (bestId === activeId) {{
      return activeId;
    }}

    const switchHysteresisPx = Math.max(10, left.clientHeight * 0.015);
    if (bestDist + switchHysteresisPx < activeDist) {{
      return bestId;
    }}
    return activeId;
  }}

  function refreshByScroll() {{
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(() => {{
      framePending = false;
      if (isProgrammatic) return;
      const nextId = nearestByCenter();
      activate(nextId, false);
      layoutCards();
    }});
  }}

  left.addEventListener("scroll", refreshByScroll, {{ passive: true }});

  function focusStep(direction) {{
    const currentIdx = Math.max(
      0,
      cardsSorted.findIndex((card) => card.dataset.annId === activeId)
    );
    const nextIdx = clamp(currentIdx + direction, 0, cardsSorted.length - 1);
    const nextId = cardsSorted[nextIdx].dataset.annId;
    if (nextId === activeId) return;
    activate(nextId, true);
  }}

  function onWheelDiscrete(event) {{
    event.preventDefault();
    if (isProgrammatic) return;

    wheelAccum += event.deltaY;
    const threshold = 42;
    if (Math.abs(wheelAccum) < threshold) return;

    if (wheelLocked) {{
      wheelAccum = 0;
      return;
    }}

    const direction = wheelAccum > 0 ? 1 : -1;
    wheelAccum = 0;
    wheelLocked = true;
    focusStep(direction);
    setTimeout(() => {{
      wheelLocked = false;
    }}, 140);
  }}

  left.addEventListener("wheel", onWheelDiscrete, {{ passive: false }});
  right.addEventListener("wheel", onWheelDiscrete, {{ passive: false }});
  rightCards.addEventListener("wheel", onWheelDiscrete, {{ passive: false }});

  cards.forEach((card) => {{
    card.style.cursor = "pointer";
    card.addEventListener("click", () => activate(card.dataset.annId, true));
  }});

  boxes.forEach((box) => {{
    box.style.cursor = "pointer";
    box.style.pointerEvents = "all";
    box.addEventListener("click", () => activate(box.dataset.annId, true));
  }});

  window.addEventListener("resize", () => {{
    layoutCards();
    activate(nearestByCenter(), false);
  }});

  layoutCards();
  activate(cardsSorted[0].dataset.annId, false);
}})();
</script>
"""
