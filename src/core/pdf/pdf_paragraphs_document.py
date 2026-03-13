import re
from dataclasses import dataclass

from src.core.pdf.pdf_lines_document import PdfLinesDocument, PdfLine

ROMAN_HEADING_RE = re.compile(r'^\s*[IVXLCDM]+\.\s+\S', re.IGNORECASE)
ARABIC_TOP_LEVEL_RE = re.compile(r'^\s*\d+\.\s+\S')


def split_by_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)
    return lines


def _has_roman_headings(lines: list[PdfLine]) -> bool:
    return any(ROMAN_HEADING_RE.match(line.text) for line in lines)


def _is_heading(line: PdfLine, has_roman: bool) -> bool:
    # Римские всегда считаем заголовком
    if ROMAN_HEADING_RE.match(line.text):
        return True

    # Арабские верхнего уровня:
    # если в тексте есть римские заголовки -> это просто пункт
    # если римских нет -> это заголовок
    if ARABIC_TOP_LEVEL_RE.match(line.text):
        return not has_roman

    return False


def _clean_heading(line: str) -> str:
    # Убираем префикс вида "I. " или "1. "
    line = re.sub(r'^\s*[IVXLCDM]+\.\s*', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^\s*\d+\.\s*', '', line)
    return line.strip()


def _make_unique_key(key: str, result: dict[str, str]) -> str:
    if key not in result:
        return key

    i = 2
    while f'{key} [{i}]' in result:
        i += 1
    return f'{key} [{i}]'


@dataclass
class PdfParagraph:
    header_line: PdfLine | None
    paragraph_lines: list[PdfLine]


@dataclass
class PdfParagraphsDocument:
    paragraphs: list[PdfParagraph]

    @staticmethod
    def from_lines_document(doc: PdfLinesDocument) -> 'PdfParagraphsDocument':
        lines = doc.lines
        has_roman = _has_roman_headings(lines)

        paragraphs = []
        current_heading = None
        buffer = []

        def flush():
            nonlocal current_heading, buffer
            if not buffer and current_heading is None:
                return

            paragraphs.append(
                PdfParagraph(
                    header_line=current_heading,
                    paragraph_lines=buffer.copy(),
                )
            )
            buffer = []

        for line in lines:
            if _is_heading(line, has_roman):
                flush()
                current_heading = line
                continue

            buffer.append(line)

        flush()
        return PdfParagraphsDocument(paragraphs=paragraphs)

# if __name__ == '__main__':
#     pages_doc = PdfPagesDocument.from_pdf('../../../data/pdf_examples/clear/TD4_clear.pdf')
#     paragraphs_doc = PdfParagraphsDocument.from_pages_document(pages_doc)
#     print(paragraphs_doc.paragraphs[1])


# def split_doc_by_headers(doc: PdfPagesDocument) -> dict[str, str]:
#     lines = doc.lines
#     has_roman = _has_roman_headings(lines)
#
#     result: dict[str, str] = {}
#     current_heading = None
#     buffer = []
#
#     def flush():
#         nonlocal current_heading, buffer
#         if current_heading is None:
#             return
#
#         key = _make_unique_key(current_heading, result)
#         result[key] = ' '.join(buffer).strip()
#         buffer = []
#
#     for line in text_lines:
#         if _is_heading(line, has_roman):
#             flush()
#             current_heading = _clean_heading(line)
#         else:
#             if current_heading is None:
#                 current_heading = 'Преамбула'
#             buffer.append(line)
#
#     flush()
#     return result


# def split_by_headers(text: str) -> dict[str, str]:
#     text_lines = split_by_lines(text)
#     has_roman = _has_roman_headings(text_lines)
#
#     result: dict[str, str] = {}
#     current_heading = None
#     buffer = []
#
#     def flush():
#         nonlocal current_heading, buffer
#         if current_heading is None:
#             return
#
#         key = _make_unique_key(current_heading, result)
#         result[key] = ' '.join(buffer).strip()
#         buffer = []
#
#     for line in text_lines:
#         if _is_heading(line, has_roman):
#             flush()
#             current_heading = _clean_heading(line)
#         else:
#             if current_heading is None:
#                 current_heading = 'Преамбула'
#             buffer.append(line)
#
#     flush()
#     return result
