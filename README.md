# P.R.A.V.O (Platform for Review and Analysis of labor contracts for Violations of Obligations)

Система для поиска соответствий и противоречий между пунктами трудового документа и нормами ТК РФ.

## Что умеет
- Разбирать PDF-документ на структурированные пункты с `bbox` координатами.
- Находить для пункта документа наиболее релевантную норму ТК РФ (retrieval по FAISS).
- Оценивать противоречие между пунктом и нормой через NLI-модель.
- Проверять обязательные поля и формальные требования трудового договора на уровне всего документа.
- Есть интерактивный UI:
  - просмотр всех страниц PDF,
  - подсветка найденных пунктов на документе,
  - карточки соответствующих норм справа,
  - синхронизированный фокус между документом и карточками,
  - чеклист проверок уровня документа со ссылками на нормы ТК РФ,
  - ручная разметка противоречий с сохранением в JSON.

## Структура `src`
- `src/core/pdf` — парсинг PDF.
- `src/core/rag` — подготовка чанков ТК и построение индекса.
- `src/core/retrieve` — поиск top-k совпадений по FAISS.
- `src/core/classification` — NLI-оценка противоречия.
- `src/core/document_checks` — проверки трудового договора на уровне всего документа.
- `src/ui` — Streamlit UI и сервисный слой анализа.

## Установка
1. Создайте и активируйте виртуальное окружение (Python 3.11).
2. Установите зависимости:

```powershell
pip install -r requirements.txt
```

## Подготовка данных (ТК и индекс)
Перед запуском UI нужно иметь построенные чанки ТК и FAISS-индекс.

Команды:

```powershell
python -m src.core.rag.chunks.build_tk_chunks
python -m src.core.rag.index.build_index 
```

## Запуск UI
```powershell
streamlit run src/ui/app.py
```

## Как работать в UI
1. Загрузите PDF-документ Трудового договора (примеры можно взять в data/pdf_examples).
2. Нажмите `Запустить анализ`.
3. Просмотрите проверки уровня документа, найденные соответствия и подсветку в документе.
4. Размечайте пары вручную (`противоречие` / `нет противоречия` / `не размечено`).
5. Нажмите `Сохранить разметку`.

Результаты разметки сохраняются в:
- `artifacts/ui_annotations/*.json`

## Примечания
- Для UI используется кэширование тяжелых моделей/индекса (FAISS + NLI), чтобы не загружать их при каждом запуске анализа.
- Для корректной работы retrieval должен существовать индекс в `artifacts/faiss_index` (или путь из `src/config.py`).

---

# English Version

P.R.A.V.O (Platform for Review and Analysis of labor contracts for Violations of Obligations) is a system for finding matches and contradictions between clauses of an employment document and provisions of the Labor Code of the Russian Federation.

## Features
- Parses a PDF document into structured clauses with `bbox` coordinates.
- Finds the most relevant Labor Code provisions for each document clause using FAISS-based retrieval.
- Scores contradictions between a document clause and a legal provision with an NLI model.
- Runs document-level checks for mandatory employment contract fields and formal requirements.
- Provides an interactive UI:
  - full PDF page viewer,
  - highlighted document clauses,
  - matching legal provision cards,
  - synchronized focus between the document and result cards,
  - document-level checklist with Labor Code references,
  - manual contradiction labeling with JSON export.

## `src` Layout
- `src/core/pdf` — PDF parsing.
- `src/core/rag` — Labor Code chunk preparation and index building.
- `src/core/retrieve` — top-k FAISS retrieval.
- `src/core/classification` — NLI contradiction scoring.
- `src/core/document_checks` — document-level employment contract checks.
- `src/ui` — Streamlit UI and analysis service layer.

## Installation
1. Create and activate a virtual environment with Python 3.11.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Data Preparation (Labor Code and Index)
Before running the UI, make sure the Labor Code chunks and FAISS index are built.

Commands:

```powershell
python -m src.core.rag.chunks.build_tk_chunks
python -m src.core.rag.index.build_index
```

## Running the UI
```powershell
streamlit run src/ui/app.py
```

## How to Use the UI
1. Upload a PDF employment contract. Example files are available in `data/pdf_examples`.
2. Click `Запустить анализ` to start the analysis.
3. Review document-level checks, matched legal provisions, and highlighted document clauses.
4. Manually label pairs as `противоречие`, `нет противоречия`, or `не размечено`.
5. Click `Сохранить разметку`.

Annotation results are saved to:
- `artifacts/ui_annotations/*.json`

## Notes
- The UI caches heavy models and indexes (FAISS + NLI) so they are not reloaded for every analysis run.
- Retrieval requires an existing index in `artifacts/faiss_index`, or the path configured in `src/config.py`.
