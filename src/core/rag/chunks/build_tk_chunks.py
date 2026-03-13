import re

import json
from pathlib import Path

from src.config import tk_path, rag_chunks_path
from src.core.rag.chunks.tk_to_json import parse_tk

ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
PUNCT_TO_KEEP_RE = re.compile(r"[^a-zа-яё0-9\s,.()]", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def extract_article_number(title: str) -> str | None:
    match = ARTICLE_RE.match(title.strip())
    return match.group(1) if match else None


def normalize_text(text: str) -> str:
    text = text.lower()
    text = PUNCT_TO_KEEP_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    if not text or not text.endswith("."):
        text += "."

    return text


def build_chunks(data: dict) -> list[dict]:
    chunks = []

    def walk(node: dict, path: list[str]) -> None:
        article_number = extract_article_number(path[-1]) if path else None

        parts = node.get("parts", [])
        if article_number is not None:
            for part in parts:
                part_number = part["part_number"]
                subparts = part.get("subparts")

                if subparts:
                    for subpart in subparts:
                        original_text = f'{part["text"]} {subpart["text"]}'
                        subpart_number = subpart["subpart_number"]

                        chunks.append({
                            "chunk_id": f"art:{article_number}:part:{part_number}:sub:{subpart_number}",
                            "normalized_text": normalize_text(original_text),
                            "original_text": original_text,
                            "article_number": article_number,
                            "part_number": part_number,
                            "subpart_number": subpart_number,
                            "hierarchy_path": path,
                        })
                else:
                    original_text = part["text"]

                    chunks.append({
                        "chunk_id": f"art:{article_number}:part:{part_number}:sub:0",
                        "normalized_text": normalize_text(original_text),
                        "original_text": original_text,
                        "article_number": article_number,
                        "part_number": part_number,
                        "subpart_number": 0,
                        "hierarchy_path": path,
                    })

        for title, child in node.items():
            if title != "parts" and isinstance(child, dict):
                walk(child, path + [title])

    for title, child in data.items():
        if isinstance(child, dict):
            walk(child, [title])

    return chunks


def save_chunks_jsonl(data: dict, output_path: str | Path) -> None:
    chunks = build_chunks(data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    data = parse_tk(tk_path)
    save_chunks_jsonl(data, rag_chunks_path)