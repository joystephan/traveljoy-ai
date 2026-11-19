# TravelJoy AI

AI-powered travel assistant that answers travel questions from real trip data. Built in layers: data analysis → RAG → agents → deployment.

## Features

### ✅ Data analysis
- Cleaned and standardized a raw dataset of 137 trips (costs, dates, destinations, transport, nationalities)
- Derived `City`, `Country`, and `Total cost` columns
- Exploratory analysis in `notebooks/01_explore_trips.ipynb`

### ✅ Structured outputs (Gemini)
- Structured outputs (Gemini) using Pydantic: the LLM returns a validated `TripRecommendation` object (city, country, best month, budget, travel style, reasons) instead of free text
- Reusable `ask()` function with automatic model fallback on transient errors (429/503)
- Model names configured in `.env`, never hardcoded

### 🚧 RAG pipeline (in progress)
- **Document ingestion:** converts each trip into a document with natural-language text for retrieval and structured metadata for filtering
- Personal data (traveler names, gender) excluded before any text reaches an LLM, verified against all 137 documents
- Missing source fields handled explicitly and never guessed
- Document shape compatible with LangChain and Chroma

## Tech Stack
- **Language:** Python
- **Data:** Pandas, NumPy, Jupyter
- **LLM:** Google Gemini via `google-genai`
- **Validation:** Pydantic
- **Config:** python-dotenv

## Project Structure
```
TravelJoy AI/
├── data/                  # raw and cleaned datasets
├── notebooks/             # exploratory analysis
├── src/
│   ├── structured_demo.py # structured outputs + ask() with model fallback
│   └── rag/
│       └── ingest.py      # trips → documents
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
python -m src.structured_demo   # structured trip recommendation from Gemini
python -m src.rag.ingest        # load trips as RAG documents
```

## Roadmap
- [x] Data analysis (Pandas, NumPy)
- [x] Structured outputs with Gemini + Pydantic
- [x] RAG document ingestion
- [ ] Retrieval + LLM answers + CLI query interface
- [ ] Embeddings + Chroma vector database
- [ ] LangGraph agent with tools + MCP
- [ ] FastAPI + Docker deployment
- [ ] Fine-tuning / multimodal

## Data & License
- **Traveler Trip Dataset** by Kiattisak Rattanaporn — [Kaggle](https://www.kaggle.com/datasets/rkiattisak/traveler-trip-data)
- Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

- **Changes made:** removed empty rows, standardized costs, dates, destinations, transport and nationality names; added City, Country, Total cost columns. Raw file: `data/trips.csv` (originally `Travel details dataset.csv`), cleaned file: `data/trips_clean.csv`.