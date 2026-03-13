import re
from dataclasses import dataclass

from src.core.pdf.pdf_lines_document import PdfLine, merge_bboxes, PdfLinesDocument
from src.core.pdf.pdf_paragraphs_document import PdfParagraph, PdfParagraphsDocument

ITEM_RE = re.compile(r'^(\d+(?:\.\d+)*\.)\s+(.*)$')


@dataclass
class PdfPoint:
    number: str | None
    text: str
    page_to_bbox: dict[int, tuple[float, float, float, float]]


@dataclass
class _PointMatch:
    line_idx: int
    path: tuple[int, ...]
    first_line_text: str


@dataclass
class _PointSegment:
    path: tuple[int, ...]
    line_start: int
    body_lines: list[PdfLine]


def _clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _join_text(*parts: str) -> str:
    cleaned = [_clean_text(part) for part in parts if _clean_text(part)]
    return ' '.join(cleaned)


def _lines_to_text(lines: list[PdfLine]) -> str:
    return _join_text(*(line.text for line in lines))


def _non_empty_lines(lines: list[PdfLine]) -> list[PdfLine]:
    return [line for line in lines if _clean_text(line.text)]


def _parse_point_number(number: str) -> tuple[int, ...]:
    return tuple(int(x) for x in number.rstrip('.').split('.'))


def _format_point_number(path: tuple[int, ...]) -> str:
    return '.'.join(str(x) for x in path) + '.'


def _parse_item_line(text: str) -> tuple[tuple[int, ...], str] | None:
    match = ITEM_RE.match(_clean_text(text))
    if not match:
        return None

    number, tail_text = match.groups()
    return _parse_point_number(number), _clean_text(tail_text)


def _extract_point_numbers(lines: list[PdfLine]) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []

    for line in lines:
        parsed = _parse_item_line(line.text)
        if parsed:
            path, _ = parsed
            result.append(path)

    return result


def _match_points(
        lines: list[PdfLine],
        points_len: int,
        prefix: tuple[int, ...] | None = None,
) -> list[_PointMatch]:
    matches: list[_PointMatch] = []

    for i, line in enumerate(lines):
        parsed = _parse_item_line(line.text)
        if not parsed:
            continue

        path, first_line_text = parsed

        if len(path) != points_len:
            continue

        if prefix is not None and path[:len(prefix)] != prefix:
            continue

        matches.append(
            _PointMatch(
                line_idx=i,
                path=path,
                first_line_text=first_line_text,
            )
        )

    return matches


def _clone_line_with_text(line: PdfLine, text: str) -> PdfLine:
    return PdfLine(
        text=_clean_text(text),
        bbox=line.bbox,
        page_num=line.page_num,
    )


def _build_segment_body_lines(
        lines: list[PdfLine],
        start_idx: int,
        end_idx: int,
        first_line_text: str,
) -> list[PdfLine]:
    result: list[PdfLine] = []

    if first_line_text:
        result.append(_clone_line_with_text(lines[start_idx], first_line_text))

    result.extend(lines[start_idx + 1:end_idx])
    return result


def _build_segments(
        lines: list[PdfLine],
        points_len: int,
        prefix: tuple[int, ...] | None = None,
) -> list[_PointSegment]:
    matches = _match_points(lines, points_len=points_len, prefix=prefix)
    segments: list[_PointSegment] = []

    for i, match in enumerate(matches):
        next_line_idx = matches[i + 1].line_idx if i + 1 < len(matches) else len(lines)

        segments.append(
            _PointSegment(
                path=match.path,
                line_start=match.line_idx,
                body_lines=_build_segment_body_lines(
                    lines=lines,
                    start_idx=match.line_idx,
                    end_idx=next_line_idx,
                    first_line_text=match.first_line_text,
                ),
            )
        )

    return segments


def merge_lines_bboxes_by_page(
        lines: list[PdfLine],
) -> dict[int, tuple[float, float, float, float]]:
    bboxes_by_page: dict[int, list[tuple[float, float, float, float]]] = {}

    for line in lines:
        bboxes_by_page.setdefault(line.page_num, []).append(line.bbox)

    return {
        page_num: merge_bboxes(page_bboxes)
        for page_num, page_bboxes in bboxes_by_page.items()
    }


def _make_point(
        path: tuple[int, ...] | None,
        text: str,
        lines: list[PdfLine],
) -> PdfPoint:
    return PdfPoint(
        number=_format_point_number(path) if path is not None else None,
        text=_clean_text(text),
        page_to_bbox=merge_lines_bboxes_by_page(lines),
    )


def _make_plain_paragraph_point(lines: list[PdfLine]) -> list[PdfPoint]:
    text = _lines_to_text(lines)
    if not text:
        return []

    return [_make_point(None, text, lines)]


def _make_preface_point(
        lines: list[PdfLine],
        first_segment_start: int,
) -> list[PdfPoint]:
    prefix_lines = lines[:first_segment_start]
    prefix_text = _lines_to_text(prefix_lines)

    if not prefix_text:
        return []

    return [_make_point(None, prefix_text, prefix_lines)]


def _flatten_leaf_segment(
        path: tuple[int, ...],
        body_lines: list[PdfLine],
        inherited_context: str,
) -> list[PdfPoint]:
    own_text = _lines_to_text(body_lines)
    full_text = _join_text(inherited_context, own_text)

    if not full_text:
        return []

    return [_make_point(path, full_text, body_lines)]


def _split_intro_and_children(
        body_lines: list[PdfLine],
        child_segments: list[_PointSegment],
) -> tuple[list[PdfLine], list[_PointSegment]]:
    intro_lines = body_lines[:child_segments[0].line_start]
    return intro_lines, child_segments


def _flatten_segment(
        body_lines: list[PdfLine],
        path: tuple[int, ...],
        inherited_context: str = '',
        inherited_lines: list[PdfLine] | None = None,
) -> list[PdfPoint]:
    inherited_lines = inherited_lines or []
    body_lines = _non_empty_lines(body_lines)

    if not body_lines:
        text = _clean_text(inherited_context)
        return [_make_point(path, text, inherited_lines)] if text else []

    child_segments = _build_segments(
        lines=body_lines,
        points_len=len(path) + 1,
        prefix=path,
    )

    if not child_segments:
        return _flatten_leaf_segment(
            path=path,
            body_lines=body_lines,
            inherited_context=inherited_context,
        )

    intro_lines, child_segments = _split_intro_and_children(body_lines, child_segments)
    current_context = _join_text(inherited_context, _lines_to_text(intro_lines))
    current_context_lines = inherited_lines + intro_lines

    result: list[PdfPoint] = []
    for child in child_segments:
        result.extend(
            _flatten_segment(
                body_lines=child.body_lines,
                path=child.path,
                inherited_context=current_context,
                inherited_lines=current_context_lines,
            )
        )

    return result


def paragraph_to_points(paragraph: PdfParagraph) -> list[PdfPoint]:
    lines = paragraph.paragraph_lines
    if not lines:
        return []

    all_point_numbers = _extract_point_numbers(lines)
    if not all_point_numbers:
        return _make_plain_paragraph_point(lines)

    top_points_len = min(len(path) for path in all_point_numbers)
    top_segments = _build_segments(lines=lines, points_len=top_points_len)

    result: list[PdfPoint] = []
    if top_segments:
        result.extend(_make_preface_point(lines, top_segments[0].line_start))

    for segment in top_segments:
        result.extend(
            _flatten_segment(
                body_lines=segment.body_lines,
                path=segment.path,
            )
        )

    return result


@dataclass
class PdfPointsDocument:
    points: list[PdfPoint]

    @staticmethod
    def from_paragraphs_document(doc: PdfParagraphsDocument) -> 'PdfPointsDocument':
        points: list[PdfPoint] = []

        for paragraph in doc.paragraphs:
            points.extend(paragraph_to_points(paragraph))

        return PdfPointsDocument(points=points)


import fitz
from PIL import Image, ImageDraw


def render_point(pdf_path, point: PdfPoint, output: str):
    doc = fitz.open(pdf_path)
    for page_num, bbox in point.page_to_bbox.items():

        page = doc[page_num]

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        draw = ImageDraw.Draw(img)

        scale = 2
        x0, y0, x1, y1 = bbox
        draw.rectangle(
            (x0 * scale, y0 * scale, x1 * scale, y1 * scale),
            outline="red",
            width=3
        )

        img.save(f'{page_num}_{output}')


if __name__ == '__main__':
    pdf_path = '../../../data/pdf_examples/clear/TD3_clear.pdf'
    pages_doc = PdfLinesDocument.from_pdf(pdf_path)
    paragraphs_doc = PdfParagraphsDocument.from_lines_document(pages_doc)
    points_doc = PdfPointsDocument.from_paragraphs_document(paragraphs_doc)
    for p in points_doc.points:
        print(p.text)
        print()
    # print(points_doc.points[11])
    # render_point(pdf_path, points_doc.points[11], 'debug_point.png')
