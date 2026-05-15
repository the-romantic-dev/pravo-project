import time
from functools import lru_cache
from collections.abc import Sequence

from transformers import pipeline, TextClassificationPipeline
from src.config import nli_model
from src.core.util.device import get_transformers_device

DEFAULT_NLI_MAX_LENGTH = 512
NLI_LABELS = ("entailment", "neutral", "contradiction")
LABEL_ALIASES = {
    "label_0": "entailment",
    "label_1": "contradiction",
    "label_2": "neutral",
}


@lru_cache(maxsize=1)
def get_nli_pipeline():
    return pipeline(
        "text-classification",
        model=nli_model,
        tokenizer=nli_model,
        top_k=None,
        device=get_transformers_device(),
    )


def get_contradiction_score(text_a: str, text_b: str, model: TextClassificationPipeline) -> float | None:
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()

    if not text_a or not text_b:
        raise ValueError("Both input strings must be non-empty")

    scores = model(
        {"text": text_a, "text_pair": text_b},
        truncation=True,
        max_length=DEFAULT_NLI_MAX_LENGTH,
    )
    for s in scores:
        if normalize_nli_label(s["label"]) == "contradiction":
            return s['score']
    return None


def get_contradiction_scores_batch(
    pairs: Sequence[tuple[str, str]],
    model: TextClassificationPipeline,
    batch_size: int = 16,
) -> list[float | None]:
    inputs = [
        {"text": (text_a or "").strip(), "text_pair": (text_b or "").strip()}
        for text_a, text_b in pairs
    ]
    if any(not item["text"] or not item["text_pair"] for item in inputs):
        raise ValueError("Both input strings must be non-empty")

    outputs = model(
        inputs,
        batch_size=batch_size,
        truncation=True,
        max_length=DEFAULT_NLI_MAX_LENGTH,
    )
    return [extract_contradiction_score(scores) for scores in outputs]


def extract_contradiction_score(scores) -> float | None:
    for score in scores:
        if normalize_nli_label(score["label"]) == "contradiction":
            return score["score"]
    return None


def normalize_nli_label(label: object) -> str:
    value = str(label).strip().lower()
    return LABEL_ALIASES.get(value, value)


def extract_nli_scores(scores) -> dict[str, float]:
    result = {label: 0.0 for label in NLI_LABELS}
    for score in scores:
        label = normalize_nli_label(score["label"])
        if label in result:
            result[label] = float(score["score"])
    return result


def get_nli_scores_batch(
    pairs: Sequence[tuple[str, str]],
    model: TextClassificationPipeline,
    batch_size: int = 16,
) -> list[dict[str, float]]:
    inputs = [
        {"text": (text_a or "").strip(), "text_pair": (text_b or "").strip()}
        for text_a, text_b in pairs
    ]
    if any(not item["text"] or not item["text_pair"] for item in inputs):
        raise ValueError("Both input strings must be non-empty")

    outputs = model(
        inputs,
        batch_size=batch_size,
        truncation=True,
        max_length=DEFAULT_NLI_MAX_LENGTH,
    )
    return [extract_nli_scores(scores) for scores in outputs]


def nli_scores(
    clf: TextClassificationPipeline,
    pairs: Sequence[tuple[str, str]],
    bidirectional: bool = True,
    batch_size: int = 16,
) -> list[dict[str, float]]:
    forward_scores = get_nli_scores_batch(pairs, clf, batch_size=batch_size)
    if not bidirectional:
        return forward_scores

    backward_pairs = [(text_b, text_a) for text_a, text_b in pairs]
    backward_scores = get_nli_scores_batch(
        backward_pairs,
        clf,
        batch_size=batch_size,
    )
    return [
        {
            label: (
                float(forward.get(label, 0.0))
                + float(backward.get(label, 0.0))
            )
            / 2.0
            for label in NLI_LABELS
        }
        for forward, backward in zip(forward_scores, backward_scores, strict=False)
    ]


def predicted_nli_label(scores: dict[str, float]) -> str:
    return max(NLI_LABELS, key=lambda label: float(scores.get(label, 0.0)))


def contradiction_score(
        clf: TextClassificationPipeline,
        text_a: str,
        text_b: str,
        bidirectional: bool = True,

) -> float:
    """
    Возвращает степень противоречия между двумя строками в диапазоне [0, 1].

    text_a, text_b: сравниваемые строки
    bidirectional=True: усредняет score для A->B и B->A
    """
    # clf = get_nli_pipeline()
    prob_ab = get_contradiction_score(text_a, text_b, clf)
    if not bidirectional:
        return prob_ab
    prob_ba = get_contradiction_score(text_b, text_a, clf)
    return (prob_ab + prob_ba) / 2


def contradiction_scores(
    clf: TextClassificationPipeline,
    pairs: Sequence[tuple[str, str]],
    bidirectional: bool = True,
    batch_size: int = 16,
) -> list[float]:
    return [
        float(scores.get("contradiction", 0.0))
        for scores in nli_scores(
            clf,
            pairs,
            bidirectional=bidirectional,
            batch_size=batch_size,
        )
    ]

if __name__ == '__main__':
    print('Start')
    start = time.perf_counter()
    clf = get_nli_pipeline()
    end = time.perf_counter()
    print(end - start)
    a = "Работник имеет право на ежегодный оплачиваемый отпуск."
    b = "Работнику предоставляется оплачиваемый отпуск."

    score = contradiction_score(clf, a, b)
    print(score)
    print(time.perf_counter() - end)
