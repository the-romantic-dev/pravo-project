import re
from dataclasses import dataclass
from functools import cached_property

from pathlib import Path

from pdftext.extraction import dictionary_output


def merge_bboxes(bboxes: list[list[float] | tuple[float, float, float, float]]
                 ) -> tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


@dataclass
class PdfLine:
    text: str
    bbox: tuple[float, float, float, float]
    page_num: int


#
# @dataclass
# class PdfPage:
#     page_num: int
#     lines: list[PdfLine]
#     bbox: tuple[float, float, float, float]
#
#     @property
#     def plain_text(self):
#         return '\n'.join([line.text for line in self.lines])


@dataclass
class PdfLinesDocument:
    lines: list[PdfLine]

    @staticmethod
    def from_pdf(pdf_path: str | Path) -> 'PdfLinesDocument':

        def _normalize_text(text: str) -> str:
            text = text.replace('\xa0', ' ')
            text = re.sub(r'_+', ' ', text)
            # text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # склейка переносов слов
            return text.strip()

        result: list[PdfLine] = []

        data = dictionary_output(pdf_path, sort=False)

        for page_data in data:
            page_num = page_data["page"]
            # page_bbox = tuple(page_data["bbox"])
            lines: list[PdfLine] = []

            for block in page_data.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    text = "".join(span.get("text", "") for span in spans)
                    bboxes = [span["bbox"] for span in spans if span.get("bbox")]

                    if not bboxes:
                        continue

                    line_bbox = merge_bboxes(bboxes)

                    lines.append(
                        PdfLine(
                            text=_normalize_text(text),
                            bbox=line_bbox,
                            page_num=page_num
                        )
                    )

            result.extend(lines)

        return PdfLinesDocument(result)

    @cached_property
    def plain_text(self) -> str:
        return '\n'.join([l.text for l in self.lines])

    # @cached_property
    # def lines(self):
    #     result = []
    #     for p in self.pages:
    #         result.extend(p.lines)
    #     return result

# if __name__ == '__main__':
#     PDF_PATH = Path('../../../data/pdf_examples/clear/TD4_clear.pdf')
#     doc = PdfDocument.from_pdf(PDF_PATH)
#     print(doc.plain_text)
