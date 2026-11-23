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
- Document shape mirrors LangChain and Chroma (`id`, `text`, `metadata`)

## Tech Stack
- **Language:** Python
- **Data:** Pandas, NumPy, Jupyter
- **LLM:** Google Gemini via `google-genai`
- **Retrieval:** keyword search with IDF weighting (pure Python)
- **Validation:** Pydantic
- **Config:** python-dotenv

## Project Structure
```
TravelJoy AI/
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
├── requirements.txt
└── README.md
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed):
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=your_main_model
GEMINI_FALLBACK_MODEL=your_fallback_model
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/).

## Usage
```bash
python -m src.structured_demo                                            # structured trip recommendation from Gemini
python -m src.rag.ingest                                                 # load trips as RAG documents
python -m src.rag.retrieve "hotel trip to London in May"                 # show the best-matching trips
python -m src.rag.answer "Which London trip had the lowest total cost?"  # grounded answer from Gemini
```

## Roadmap
- [x] Data analysis (Pandas, NumPy)
- [x] Structured outputs with Gemini + Pydantic
- [x] RAG document ingestion
- [x] Keyword retrieval + grounded LLM answers
- [ ] CLI query interface
- [ ] Embeddings + Chroma vector database
- [ ] LangGraph agent with tools + MCP
- [ ] FastAPI + Docker deployment
- [ ] Fine-tuning / multimodal

## Data & License
- **Traveler Trip Dataset** by Kiattisak Rattanaporn — [Kaggle](https://www.kaggle.com/datasets/rkiattisak/traveler-trip-data)
- Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Changes made:** removed empty rows, standardized costs, dates, destinations, transport and nationality names; added City, Country, Total cost columns. Raw file: `data/trips.csv` (originally `Travel details dataset.csv`), cleaned file: `data/trips_clean.csv`. For the RAG documents, traveler names, gender, and nationality are removed and ages are generalized to age groups.