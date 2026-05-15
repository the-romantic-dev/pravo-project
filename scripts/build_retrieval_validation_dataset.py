from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
TK_CHUNKS_PATH = PROJECT_DIR / "artifacts" / "tk_chunks_new.jsonl"
ANNOTATION_PATHS = [
    PROJECT_DIR / "data" / "test_chunks_annotated.jsonl",
    PROJECT_DIR / "data" / "test_chunks2_annotated.jsonl",
]
OUT_JSONL = PROJECT_DIR / "data" / "retrieval_validation_pairs.jsonl"
OUT_CSV = PROJECT_DIR / "data" / "retrieval_validation_pairs.csv"

MAX_ANNOTATED_POSITIVES = 110
NEGATIVE_PAIRS = 70
RANDOM_SEED = 42


CSV_FIELDS = [
    "pair_id",
    "source",
    "contract_heading",
    "contract_point_number",
    "contract_text",
    "tk_chunk_id",
    "tk_article_number",
    "tk_part_number",
    "tk_subpart_number",
    "tk_norm_text",
    "has_relation",
    "has_contradiction",
    "comment",
]


SYNTHETIC_CONTRADICTIONS = [
    (
        "Работник обязан выполнять порученную работу под угрозой штрафа и иных мер воздействия со стороны работодателя.",
        "art:4:part:1:sub:0",
        "Договор допускает принудительное выполнение работы, тогда как ТК прямо запрещает принудительный труд.",
    ),
    (
        "Для поддержания дисциплины работодатель вправе заставлять работника выполнять обязанности под угрозой наказания.",
        "art:4:part:2:sub:1",
        "Пункт описывает работу под угрозой наказания как допустимую меру дисциплины; норма относит это к принудительному труду.",
    ),
    (
        "Работник не вправе отказаться от выполнения работы, даже если она угрожает его жизни или здоровью.",
        "art:4:part:3:sub:2",
        "Пункт исключает право отказаться от опасной работы, тогда как норма относит принуждение к такой работе к принудительному труду.",
    ),
    (
        "Нормальная продолжительность рабочего времени работника составляет 60 часов в неделю.",
        "art:91:part:2:sub:0",
        "Пункт превышает установленный законом предел 40 часов в неделю.",
    ),
    (
        "Работнику устанавливается шестидневная рабочая неделя общей продолжительностью 48 часов.",
        "art:91:part:2:sub:0",
        "Пункт устанавливает норму рабочего времени выше 40 часов в неделю.",
    ),
    (
        "Работодатель не обязан вести учет фактически отработанного работником времени.",
        "art:91:part:4:sub:0",
        "Пункт снимает с работодателя обязанность вести учет рабочего времени.",
    ),
    (
        "Работник может привлекаться к сверхурочной работе без письменного согласия в любых производственных ситуациях.",
        "art:99:part:4:sub:0",
        "Пункт разрешает сверхурочную работу без согласия шире, чем допускает ТК.",
    ),
    (
        "Сверхурочная работа может составлять до 10 часов в течение двух дней подряд и до 300 часов в год.",
        "art:99:part:5:sub:0",
        "Пункт превышает лимиты сверхурочной работы.",
    ),
    (
        "Работодатель не ведет отдельный учет продолжительности сверхурочной работы.",
        "art:99:part:6:sub:0",
        "Пункт противоречит обязанности точно учитывать сверхурочные часы.",
    ),
    (
        "Перерыв для отдыха и питания составляет 10 минут и в рабочее время не включается.",
        "art:108:part:1:sub:0",
        "Пункт устанавливает перерыв меньше минимальных 30 минут.",
    ),
    (
        "Еженедельный непрерывный отдых работника составляет не менее 24 часов.",
        "art:110:part:1:sub:0",
        "Пункт устанавливает еженедельный отдых меньше 42 часов.",
    ),
    (
        "Работник обязан выходить на работу в выходные и праздничные дни по требованию работодателя без письменного согласия.",
        "art:113:part:1:sub:0",
        "Пункт делает работу в выходные обычной обязанностью, хотя она запрещена кроме случаев ТК.",
    ),
    (
        "Привлечение к работе в выходной день производится устным распоряжением непосредственного руководителя.",
        "art:113:part:7:sub:0",
        "Пункт отменяет письменное распоряжение работодателя для работы в выходные и праздники.",
    ),
    (
        "Ежегодный основной оплачиваемый отпуск работника составляет 14 календарных дней.",
        "art:115:part:1:sub:0",
        "Пункт уменьшает ежегодный основной оплачиваемый отпуск ниже 28 календарных дней.",
    ),
    (
        "Работодатель предоставляет инвалиду ежегодный основной отпуск продолжительностью 28 календарных дней.",
        "art:115:part:3:sub:0",
        "Для работников-инвалидов отпуск должен быть не менее 30 календарных дней.",
    ),
    (
        "Ежегодный отпуск может быть разделен на части любой продолжительности, включая части по 7 календарных дней.",
        "art:125:part:1:sub:0",
        "Пункт не гарантирует хотя бы одну часть отпуска не менее 14 календарных дней.",
    ),
    (
        "Работодатель вправе отозвать работника из отпуска без его согласия.",
        "art:125:part:2:sub:0",
        "Пункт допускает отзыв из отпуска без согласия работника.",
    ),
    (
        "Беременная работница может быть отозвана из ежегодного отпуска по распоряжению работодателя.",
        "art:125:part:3:sub:0",
        "Пункт допускает отзыв беременной работницы из отпуска, что запрещено.",
    ),
    (
        "Заработная плата выплачивается один раз в месяц до 30 числа месяца, следующего за расчетным.",
        "art:136:part:6:sub:0",
        "Пункт нарушает правило выплаты зарплаты не реже чем каждые полмесяца.",
    ),
    (
        "Если день выплаты зарплаты совпадает с выходным, выплата переносится на ближайший следующий рабочий день.",
        "art:136:part:8:sub:0",
        "Пункт переносит выплату после выходного, хотя она должна производиться накануне.",
    ),
    (
        "Оплата отпуска производится в течение трех дней после начала отпуска.",
        "art:136:part:9:sub:0",
        "Пункт сдвигает оплату отпуска позднее установленного срока до начала отпуска.",
    ),
    (
        "При увольнении все суммы выплачиваются работнику в течение 30 календарных дней после прекращения договора.",
        "art:140:part:1:sub:0",
        "Пункт нарушает срок окончательного расчета при увольнении.",
    ),
    (
        "При задержке заработной платы компенсация работнику не начисляется, если задержка произошла без вины работодателя.",
        "art:236:part:2:sub:0",
        "Пункт связывает компенсацию с виной работодателя, хотя обязанность возникает независимо от вины.",
    ),
    (
        "За задержку заработной платы работодатель выплачивает только сумму долга без процентов.",
        "art:236:part:1:sub:0",
        "Пункт исключает денежную компенсацию за задержку выплат.",
    ),
    (
        "Испытательный срок для обычного работника устанавливается продолжительностью 12 месяцев.",
        "art:70:part:5:sub:0",
        "Пункт превышает максимальный срок испытания для обычного работника.",
    ),
    (
        "Беременной работнице устанавливается испытательный срок три месяца.",
        "art:70:part:4:sub:2",
        "Пункт устанавливает испытание беременной женщине, что запрещено.",
    ),
    (
        "Работнику младше восемнадцати лет устанавливается испытание при приеме на работу.",
        "art:70:part:4:sub:3",
        "Пункт устанавливает испытание несовершеннолетнему работнику.",
    ),
    (
        "При договоре на четыре месяца испытательный срок составляет один месяц.",
        "art:70:part:6:sub:0",
        "Для договора от двух до шести месяцев испытание не может превышать двух недель.",
    ),
    (
        "Работник может уволиться по собственному желанию только предупредив работодателя за шесть месяцев.",
        "art:80:part:1:sub:0",
        "Пункт ухудшает право работника на увольнение с предупреждением за две недели.",
    ),
    (
        "До истечения срока предупреждения об увольнении работник не вправе отозвать заявление.",
        "art:80:part:4:sub:0",
        "Пункт лишает работника права отозвать заявление об увольнении.",
    ),
    (
        "В последний день работы работодатель не обязан выдавать работнику трудовую книжку или сведения о трудовой деятельности.",
        "art:80:part:5:sub:0",
        "Пункт отменяет обязанность работодателя выдать документы и произвести расчет в последний день работы.",
    ),
    (
        "Трудовой договор заключается устно, выдача работнику экземпляра договора не требуется.",
        "art:67:part:1:sub:0",
        "Пункт противоречит письменной форме договора и передаче экземпляра работнику.",
    ),
    (
        "Если работник фактически допущен к работе, работодатель вправе не оформлять трудовой договор письменно.",
        "art:67:part:2:sub:0",
        "Пункт отменяет обязанность оформить договор при фактическом допуске к работе.",
    ),
    (
        "Срочный трудовой договор может быть заключен на срок десять лет.",
        "art:58:part:1:sub:2",
        "Пункт превышает общий пятилетний предел срочного трудового договора.",
    ),
    (
        "Срочный договор заключается для того, чтобы не предоставлять работнику гарантии бессрочного договора.",
        "art:58:part:6:sub:0",
        "Пункт прямо предусматривает запрещенную цель срочного договора.",
    ),
    (
        "Месячная заработная плата полностью отработавшего норму работника составляет 5 000 рублей.",
        "art:133:part:3:sub:0",
        "Пункт допускает оплату ниже минимального размера оплаты труда.",
    ),
    (
        "При направлении в командировку за работником не сохраняется средний заработок.",
        "art:167:part:1:sub:0",
        "Пункт исключает гарантию среднего заработка при командировке.",
    ),
    (
        "Расходы, связанные со служебной командировкой, работнику не возмещаются.",
        "art:167:part:1:sub:0",
        "Пункт отменяет возмещение командировочных расходов.",
    ),
    (
        "При временной нетрудоспособности пособие работнику не выплачивается.",
        "art:183:part:1:sub:0",
        "Пункт исключает выплату пособия по временной нетрудоспособности.",
    ),
    (
        "Работа по совместительству допускается для работников младше восемнадцати лет.",
        "art:282:part:5:sub:0",
        "Пункт допускает совместительство несовершеннолетнего, что ТК не допускает.",
    ),
    (
        "На дистанционных работников трудовое законодательство в период удаленной работы не распространяется.",
        "art:312.1:part:4:sub:0",
        "Пункт исключает действие трудового законодательства для дистанционных работников.",
    ),
]


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunks() -> dict[str, dict[str, Any]]:
    chunks = load_jsonl(TK_CHUNKS_PATH)
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def row_from_chunk(
    *,
    pair_id: str,
    source: str,
    contract_text: str,
    chunk: dict[str, Any],
    has_relation: int,
    has_contradiction: int,
    comment: str,
    contract_heading: str = "",
    contract_point_number: str = "",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "source": source,
        "contract_heading": clean_text(contract_heading),
        "contract_point_number": clean_text(contract_point_number),
        "contract_text": clean_text(contract_text),
        "tk_chunk_id": chunk["chunk_id"],
        "tk_article_number": str(chunk.get("article_number", "")),
        "tk_part_number": str(chunk.get("part_number", "")),
        "tk_subpart_number": str(chunk.get("subpart_number", "")),
        "tk_norm_text": clean_text(chunk.get("original_text") or chunk.get("normalized_text")),
        "has_relation": has_relation,
        "has_contradiction": has_contradiction,
        "comment": clean_text(comment),
    }


def annotated_positive_rows(chunks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    weak_comment_markers = (
        "косвенно",
        "косвенное",
        "может иметь значение",
        "если бы",
        "не выбран",
        "форма отдельно оговаривает",
        "альтернативу",
        "подразумевается",
    )

    for path in ANNOTATION_PATHS:
        for item in load_jsonl(path):
            contract_text = clean_text(item.get("document_point"))
            if not contract_text:
                continue
            for match in item.get("matched_tk_norms", []):
                comment = clean_text(match.get("comment"))
                comment_lower = comment.lower()
                if any(marker in comment_lower for marker in weak_comment_markers):
                    continue
                chunk_id = match.get("chunk_id")
                if chunk_id not in chunks_by_id:
                    continue
                key = (contract_text, chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    row_from_chunk(
                        pair_id=f"rel-{len(rows) + 1:04d}",
                        source=f"annotated:{path.name}",
                        contract_heading=item.get("heading", ""),
                        contract_point_number=item.get("point_number", ""),
                        contract_text=contract_text,
                        chunk=chunks_by_id[chunk_id],
                        has_relation=1,
                        has_contradiction=0,
                        comment=comment or "Аннотированная релевантная пара из примеров договоров.",
                    )
                )

    return rows[:MAX_ANNOTATED_POSITIVES]


def synthetic_contradiction_rows(chunks_by_id: dict[str, dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    rows = []
    for idx, (contract_text, chunk_id, comment) in enumerate(SYNTHETIC_CONTRADICTIONS, start=start_index):
        if chunk_id not in chunks_by_id:
            raise KeyError(f"Unknown TK chunk id in synthetic case: {chunk_id}")
        rows.append(
            row_from_chunk(
                pair_id=f"contr-{idx:04d}",
                source="synthetic_contradiction",
                contract_text=contract_text,
                chunk=chunks_by_id[chunk_id],
                has_relation=1,
                has_contradiction=1,
                comment=comment,
            )
        )
    return rows


def negative_rows(
    chunks_by_id: dict[str, dict[str, Any]],
    positive_rows: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    chunks = list(chunks_by_id.values())
    negatives: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()

    def article_as_float(chunk: dict[str, Any]) -> float | None:
        article = str(chunk.get("article_number", ""))
        try:
            return float(article)
        except ValueError:
            return None

    def is_distant_negative_topic(chunk: dict[str, Any]) -> bool:
        article = article_as_float(chunk)
        if article is None:
            return False
        return (
            23 <= article <= 55
            or 352 <= article <= 424
            or 327 <= article < 328
            or 348 <= article < 349
        )

    preferred_chunks = [
        chunk
        for chunk in chunks
        if is_distant_negative_topic(chunk)
    ]

    shuffled_positive_rows = positive_rows[:]
    rng.shuffle(shuffled_positive_rows)

    for source_row in shuffled_positive_rows:
        source_article = source_row["tk_article_number"]
        contract_text = source_row["contract_text"]
        candidates = [
            chunk
            for chunk in preferred_chunks
            if str(chunk.get("article_number")) != source_article
            and chunk["chunk_id"] != source_row["tk_chunk_id"]
        ]
        rng.shuffle(candidates)
        for chunk in candidates:
            key = (contract_text, chunk["chunk_id"])
            if key in used:
                continue
            used.add(key)
            negatives.append(
                row_from_chunk(
                    pair_id=f"neg-{start_index + len(negatives):04d}",
                    source="synthetic_hard_negative_from_contract_examples",
                    contract_heading=source_row["contract_heading"],
                    contract_point_number=source_row["contract_point_number"],
                    contract_text=contract_text,
                    chunk=chunk,
                    has_relation=0,
                    has_contradiction=0,
                    comment=(
                        "Нерелевантная пара: пункт договора взят из примеров, "
                        "а норма ТК подобрана из другой темы для проверки отсечения ложных совпадений."
                    ),
                )
            )
            break
        if len(negatives) >= NEGATIVE_PAIRS:
            break

    return negatives


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks_by_id = load_chunks()
    positives = annotated_positive_rows(chunks_by_id)
    contradictions = synthetic_contradiction_rows(chunks_by_id, start_index=1)
    negatives = negative_rows(chunks_by_id, positives, start_index=1)
    rows = positives + contradictions + negatives
    write_outputs(rows)

    relation_count = sum(row["has_relation"] for row in rows)
    contradiction_count = sum(row["has_contradiction"] for row in rows)
    print(f"Wrote {len(rows)} rows")
    print(f"has_relation=1: {relation_count}")
    print(f"has_contradiction=1: {contradiction_count}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
