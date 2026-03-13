import re
from dataclasses import dataclass
from pathlib import Path

from src.core.pdf.pdf_lines_document import PdfLinesDocument
from src.core.pdf.pdf_paragraphs_document import PdfParagraphsDocument, PdfParagraph
from src.core.pdf.pdf_points_document import PdfPointsDocument, PdfPoint


BBox = tuple[float, float, float, float]


@dataclass
class PdfChunk:
    heading: str
    point_number: str | None
    text: str
    normalized_text: str
    page_to_bbox: dict[int, BBox]


@dataclass
class PdfChunksDocument:
    chunks: list[PdfChunk]

    def by_heading(self) -> dict[str, list[PdfChunk]]:
        result: dict[str, list[PdfChunk]] = {}
        for chunk in self.chunks:
            result.setdefault(chunk.heading, []).append(chunk)
        return result


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\.,]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_heading_text(text: str) -> str:
    text = re.sub(r'^\s*[IVXLCDM]+\.\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\.\s*', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def paragraph_heading(paragraph: PdfParagraph) -> str:
    if paragraph.header_line is None:
        return 'Преамбула'
    return clean_heading_text(paragraph.header_line.text)


def remove_signature_tail(text: str) -> str:
    chunks = re.findall(r'[^.]*?(?:\.|$)', text, flags=re.S)

    for i, chunk in enumerate(chunks):
        chunk_clean = chunk.strip()
        if not chunk_clean:
            continue

        if re.search(r'\bподпись\b', chunk_clean, flags=re.IGNORECASE):
            return re.sub(r'\s+', ' ', ''.join(chunks[:i]).strip())

    return text


def is_service_heading(heading: str) -> bool:
    heading_norm = normalize_text(heading)
    return heading_norm in {
        'адреса, реквизиты и подписи сторон',
        'юридические адреса и платёжные реквизиты сторон',
        'подписи сторон',
    }


def should_skip_heading(heading: str) -> bool:
    return normalize_text(heading) == 'преамбула' or is_service_heading(heading)


def cleanup_chunks(chunks: list[PdfChunk]) -> list[PdfChunk]:
    result: list[PdfChunk] = []

    for chunk in chunks:
        if should_skip_heading(chunk.heading):
            continue

        if normalize_text(chunk.heading) == 'заключительные положения':
            cleaned_text = remove_signature_tail(chunk.text)
            chunk = PdfChunk(
                heading=chunk.heading,
                point_number=chunk.point_number,
                text=cleaned_text,
                normalized_text=normalize_text(cleaned_text),
                page_to_bbox=chunk.page_to_bbox,
            )

            if not chunk.normalized_text:
                continue

        result.append(chunk)

    return result


def paragraph_to_chunks(paragraph: PdfParagraph) -> list[PdfChunk]:
    heading = paragraph_heading(paragraph)
    points = PdfPointsDocument.from_paragraphs_document(
        PdfParagraphsDocument(paragraphs=[paragraph])
    ).points

    result: list[PdfChunk] = []
    for point in points:
        if not point.text.strip():
            continue

        result.append(
            PdfChunk(
                heading=heading,
                point_number=point.number,
                text=point.text,
                normalized_text=normalize_text(point.text),
                page_to_bbox=point.page_to_bbox,
            )
        )

    return result


def pdf_to_chunks_document(pdf_path: str | Path) -> PdfChunksDocument:
    lines_doc = PdfLinesDocument.from_pdf(pdf_path)
    paragraphs_doc = PdfParagraphsDocument.from_lines_document(lines_doc)

    chunks: list[PdfChunk] = []
    for paragraph in paragraphs_doc.paragraphs:
        chunks.extend(paragraph_to_chunks(paragraph))

    chunks = cleanup_chunks(chunks)
    return PdfChunksDocument(chunks=chunks)


def pdf_to_chunks_by_heading(pdf_path: str | Path) -> dict[str, list[PdfChunk]]:
    return pdf_to_chunks_document(pdf_path).by_heading()


if __name__ == '__main__':
    pdf_path = Path('../../../data/pdf_examples/clear/TD4_clear.pdf')
    chunks_doc = pdf_to_chunks_document(pdf_path)

    # for chunk in chunks_doc.chunks[-1]:
    print(chunks_doc.chunks[-1])