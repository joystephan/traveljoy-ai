# TravelJoy AI

AI-powered travel assistant that answers travel questions from real trip data. Built in layers: data analysis → RAG → agents → deployment.

## Features

### ✅ Data analysis
- Cleaned and standardized a raw dataset of 139 records into 137 trips (costs, dates, destinations, transport, nationalities)
- Derived `City`, `Country`, and `Total cost` columns
- Exploratory analysis in `notebooks/01_explore_trips.ipynb`

### ✅ Structured outputs (Gemini)
- Structured outputs (Gemini) using Pydantic: the LLM returns a validated `TripRecommendation` object (city, country, best month, budget, travel style, reasons) instead of free text

### ✅ Gemini client
- Shared client (`src/llm.py`) used by every feature: `generate_text()` and `generate_structured()`
- Automatic model fallback on transient errors (429, 500, 503, 504)
- 30-second request timeout, so a slow or unresponsive model falls through to the fallback
- Model names configured in `.env`, never hardcoded; the API key is loaded from `.env` and never included in prompts or in the project's log messages

### 🚧 RAG pipeline (in progress)
- **Document ingestion:** converts each trip into a document with natural-language text for retrieval and structured metadata for filtering
- **Privacy by design:** traveler names, gender, and nationality are excluded and age is generalized to an age group (e.g. "30s") before any text reaches an LLM; names and nationalities verified absent from all 137 documents
- **Keyword retrieval:** ranks trips by shared terms weighted by inverse document frequency (IDF), so rare terms like a city name outweigh common ones
- **Grounded answers:** retrieved trips are sent to Gemini as context; the model is instructed to cite trip IDs (e.g. `[trip-89]`) and to say so when the data can't answer; no model call is made when no trips match
- Missing source fields handled explicitly and never guessed
- Dataset path resolved from the module's location, so ingestion works from any working directory
- Document shape mirrors LangChain and Chroma (`id`, `text`, `metadata`)

### ✅ Command-line interface
- Interactive question loop (`main.py`): prints each grounded answer followed by the IDs of the trips retrieved for it
- Shows example questions on startup
- `--top-k` option sets how many trips are retrieved per question (default: 5)
- Exits cleanly on `quit`, `exit`, an empty line, or Ctrl+C
- Shows a friendly message instead of a traceback when every configured model is unavailable, and keeps the session running
- Logs model fallback events from the project's Gemini client while keeping third-party libraries at warning level

### ✅ Automated tests
- 28 pytest tests covering document ingestion, retrieval, and grounded answers
- **Privacy checks:** identity columns are dropped, and no traveler name or nationality appears in any document's text
- **Data handling:** no missing values in document text or metadata; trips with missing transport or no city are described without guessed values
- **Retrieval:** IDF ranking order, case-insensitive matching, the top-k limit, and empty results when no trip matches
- **Grounded answers:** prompt format, no model call when no trips match, the system instruction is sent with every model call, and model errors are passed to the caller
- The Gemini call is replaced with a fake in tests, so the suite runs offline, costs nothing, and never calls the real API

## Tech Stack
- **Language:** Python
- **Data:** Pandas, NumPy, Jupyter
- **LLM:** Google Gemini via `google-genai`
- **Retrieval:** keyword search with IDF weighting (pure Python)
- **Validation:** Pydantic
- **Config:** python-dotenv
- **Testing:** pytest
- **Code quality:** Ruff (linting and formatting)

## Project Structure
```
traveljoy-ai/
├── data/                  # raw and cleaned datasets
├── notebooks/             # exploratory analysis
├── src/
│   ├── llm.py             # Gemini client: timeout + model fallback
│   ├── structured_demo.py # structured trip recommendation
│   └── rag/
│       ├── ingest.py      # trips → documents
│       ├── retrieve.py    # keyword retrieval (IDF)
│       └── answer.py      # retrieved trips + question → grounded answer
├── tests/
│   ├── test_ingest.py     # privacy filtering, missing data, age groups
│   ├── test_retrieve.py   # IDF ranking, empty results, top-k
│   └── test_answer.py     # prompt building, no-match guard (fake Gemini)
├── main.py                # CLI: ask travel questions in a loop
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone https://github.com/joystephan/traveljoy-ai.git
cd traveljoy-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `python -m venv .venv` and `.venv\Scripts\activate` instead.

All dependency versions are pinned in `requirements.txt`.

Create a `.env` file in the project root (never committed):
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=your_main_model
GEMINI_FALLBACK_MODEL=your_fallback_model
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/).

## Usage
```bash
python main.py                                                           # interactive Q&A over the trip dataset
python main.py --top-k 3                                                 # retrieve 3 trips per question instead of 5
python main.py --help                                                    # list CLI options
python -m src.structured_demo                                            # structured trip recommendation from Gemini
python -m src.rag.ingest                                                 # load trips as RAG documents
python -m src.rag.retrieve "hotel trip to London in May"                 # show the best-matching trips
python -m src.rag.answer "Which London trip had the lowest total cost?"  # grounded answer from Gemini
```

## Testing
Run from the project root:
```bash
python -m pytest      # run all tests
python -m pytest -v   # list each test by name
```

Tests don't need an API key or network access.

## Code quality
Code is formatted and linted with Ruff:
```bash
ruff check main.py src tests            # lint
ruff format --check main.py src tests   # verify formatting
```

## Roadmap
- [x] Data analysis (Pandas, NumPy)
- [x] Structured outputs with Gemini + Pydantic
- [x] RAG document ingestion
- [x] Keyword retrieval + grounded LLM answers
- [x] CLI query interface
- [x] Automated tests (pytest)
- [ ] Embeddings + Chroma vector database
- [ ] LangGraph agent with tools + MCP
- [ ] FastAPI + Docker deployment
- [ ] Fine-tuning / multimodal

## Data & License
- **Traveler Trip Dataset** by Kiattisak Rattanaporn — [Kaggle](https://www.kaggle.com/datasets/rkiattisak/traveler-trip-data)
- Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made:** removed empty rows, standardized costs, dates, destinations, transport and nationality names; added City, Country, Total cost columns. Raw file: `data/trips.csv` (originally `Travel details dataset.csv`), cleaned file: `data/trips_clean.csv`. For the RAG documents, traveler names, gender, and nationality are removed and ages are generalized to age groups.
