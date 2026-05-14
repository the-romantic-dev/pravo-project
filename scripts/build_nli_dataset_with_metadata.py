from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PAIRS_JSONL = PROJECT_DIR / "data" / "retrieval_validation_pairs.jsonl"
TK_CHUNKS_JSONL = PROJECT_DIR / "artifacts" / "tk_chunks_new.jsonl"
OUT_JSONL = PROJECT_DIR / "data" / "nli_dataset_with_metadata.jsonl"
OUT_CSV = PROJECT_DIR / "data" / "nli_dataset_with_metadata.csv"


CSV_FIELDS = [
    "pair_id",
    "premise",
    "premise_with_article_title",
    "premise_with_chapter",
    "premise_with_full_context",
    "hypothesis",
    "label",
    "source",
    "contract_heading",
    "contract_point_number",
    "tk_chunk_id",
    "tk_article_number",
    "tk_part_number",
    "tk_subpart_number",
    "code_part_title",
    "section_title",
    "chapter_title",
    "article_title",
    "hierarchy_path",
    "has_relation",
    "has_contradiction",
    "comment",
]


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_chunks_by_id() -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in load_jsonl(TK_CHUNKS_JSONL)}


def label_from_flags(has_relation: Any, has_contradiction: Any) -> str:
    if int(has_contradiction) == 1:
        return "contradiction"
    if int(has_relation) == 1:
        return "entailment"
    return "neutral"


def split_hierarchy(hierarchy_path: list[Any]) -> dict[str, str]:
    items = [clean_text(item) for item in hierarchy_path if clean_text(item)]
    return {
        "code_part_title": items[0] if len(items) > 0 else "",
        "section_title": items[1] if len(items) > 1 else "",
        "chapter_title": items[2] if len(items) > 2 else "",
        "article_title": items[3] if len(items) > 3 else "",
    }


def join_context(*parts: str) -> str:
    return clean_text(". ".join(part.strip().rstrip(".") for part in parts if part.strip()))


def build_row(pair: dict[str, Any], chunk: dict[str, Any] | None) -> dict[str, Any]:
    hierarchy_path = list(chunk.get("hierarchy_path", [])) if chunk else []
    hierarchy = split_hierarchy(hierarchy_path)
    premise = clean_text(pair.get("tk_norm_text"))
    hypothesis = clean_text(pair.get("contract_text"))

    return {
        "pair_id": clean_text(pair.get("pair_id")),
        "premise": premise,
        "premise_with_article_title": join_context(
            hierarchy["article_title"],
            premise,
        ),
        "premise_with_chapter": join_context(
            hierarchy["chapter_title"],
            hierarchy["article_title"],
            premise,
        ),
        "premise_with_full_context": join_context(
            hierarchy["code_part_title"],
            hierarchy["section_title"],
            hierarchy["chapter_title"],
            hierarchy["article_title"],
            premise,
        ),
        "hypothesis": hypothesis,
        "label": label_from_flags(pair.get("has_relation", 0), pair.get("has_contradiction", 0)),
        "source": clean_text(pair.get("source")),
        "contract_heading": clean_text(pair.get("contract_heading")),
        "contract_point_number": clean_text(pair.get("contract_point_number")),
        "tk_chunk_id": clean_text(pair.get("tk_chunk_id")),
        "tk_article_number": clean_text(pair.get("tk_article_number")),
        "tk_part_number": clean_text(pair.get("tk_part_number")),
        "tk_subpart_number": clean_text(pair.get("tk_subpart_number")),
        "code_part_title": hierarchy["code_part_title"],
        "section_title": hierarchy["section_title"],
        "chapter_title": hierarchy["chapter_title"],
        "article_title": hierarchy["article_title"],
        "hierarchy_path": " > ".join(clean_text(item) for item in hierarchy_path),
        "has_relation": int(pair.get("has_relation", 0)),
        "has_contradiction": int(pair.get("has_contradiction", 0)),
        "comment": clean_text(pair.get("comment")),
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks_by_id = load_chunks_by_id()
    pairs = load_jsonl(INPUT_PAIRS_JSONL)
    rows = [
        build_row(pair, chunks_by_id.get(clean_text(pair.get("tk_chunk_id"))))
        for pair in pairs
    ]

    write_jsonl(rows, OUT_JSONL)
    write_csv(rows, OUT_CSV)

    labels = {label: 0 for label in ("entailment", "contradiction", "neutral")}
    for row in rows:
        labels[row["label"]] += 1

    print(f"Wrote {len(rows)} rows")
    print(f"Labels: {labels}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
