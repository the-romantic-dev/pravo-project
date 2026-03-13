import re
from docx import Document
import json

SKIP_RE = re.compile(
    r"\([^)]*Федерал\w*[^)]*закон\w*[^)]*\)",
    re.IGNORECASE,
)

SKIP_PHRASES = (
    "признать утратившими силу",
    "закон российской федерации от",
    "федеральный закон от",
    'рсфср от',
    'федерального закона от',
    'президент.',
    'российской федерации.'
)

SKIP_PARTS = (
    "российской федерации",
    'в.путин',
    'москва, кремль',
    '30 декабря 2001 года',
    'n 197-фз',
    'президент'
)
def get_outline_level(paragraph) -> int | None:
    try:
        lvl = paragraph._p.pPr.outlineLvl
        return int(lvl.val) + 1 if lvl is not None else None
    except Exception:
        return None


def should_skip(text: str) -> bool:
    lowered = text.lower()
    return bool(SKIP_RE.search(text)) or any(phrase in lowered for phrase in SKIP_PHRASES) or any(part == lowered for part in SKIP_PARTS)

def parse_parts(lines: list[str]) -> list[dict]:
    parts = []
    i = 0
    part_number = 1

    while i < len(lines):
        text = lines[i].strip()
        i += 1

        if not text or should_skip(text):
            continue

        part = {
            "text": text,
            "part_number": part_number,
        }
        part_number += 1

        if text.endswith(":"):
            subparts = []
            subpart_number = 1

            while i < len(lines):
                subtext = lines[i].strip()

                if not subtext:
                    i += 1
                    continue

                if should_skip(subtext):
                    i += 1
                    continue

                if not subtext.endswith((";", ".")):
                    break

                subparts.append({
                    "text": subtext,
                    "subpart_number": subpart_number,
                })
                subpart_number += 1
                i += 1

                if subtext.endswith("."):
                    break

            if subparts:
                part["subparts"] = subparts

        parts.append(part)

    return parts


def parse_tk(docx_path: str) -> dict:
    doc = Document(docx_path)
    root = {}
    stack: list[tuple[int, dict]] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        level = get_outline_level(paragraph)

        if level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent = stack[-1][1] if stack else root
            node = parent.setdefault(text, {})
            stack.append((level, node))
            continue

        if stack:
            stack[-1][1].setdefault("_lines", []).append(text)

    def finalize(node: dict) -> None:
        lines = node.pop("_lines", [])
        parts = parse_parts(lines)
        if parts:
            node["parts"] = parts

        for value in node.values():
            if isinstance(value, dict):
                finalize(value)

    for node in root.values():
        finalize(node)

    return root


def save_json(data, json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# if __name__ == "__main__":
#     doc_path = r'D:\Projects\TKRF_Course_Work\data\Трудовой_кодекс_Российской_Федерации_от_30_12_2001_N_197_ФЗ.docx'
#     data = extract_headings(doc_path)
#     # save_json(data, "headings.json")
#     print(data['ЧАСТЬ ПЕРВАЯ']['Раздел I. ОБЩИЕ ПОЛОЖЕНИЯ']['Глава 1. ОСНОВНЫЕ НАЧАЛА ТРУДОВОГО ЗАКОНОДАТЕЛЬСТВА'][
#               'Статья 4. Запрещение принудительного труда'])
