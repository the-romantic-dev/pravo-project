import time
from functools import lru_cache
from transformers import pipeline, TextClassificationPipeline
from src.config import nli_model


@lru_cache(maxsize=1)
def get_nli_pipeline():
    return pipeline(
        "text-classification",
        model=nli_model,
        tokenizer=nli_model,
        top_k=None
    )


def get_contradiction_score(text_a: str, text_b: str, model: TextClassificationPipeline) -> float | None:
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()

    if not text_a or not text_b:
        raise ValueError("Both input strings must be non-empty")

    scores = model({"text": text_a, "text_pair": text_b})
    for s in scores:
        if s['label'] == 'contradiction':
            return s['score']
    return None


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
