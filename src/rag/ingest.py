"""Load trip records and convert them into documents for the RAG pipeline.

Each row of the cleaned trips dataset becomes one Document: a natural-language
description used for retrieval and prompting, plus structured metadata used
for filtering. Personally identifying columns are excluded, and fields missing
from the source data are omitted rather than guessed.

Usage:
    python -m src.rag.ingest
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TRIPS_CSV_PATH = Path("data/trips_clean.csv")

# Names and gender add no value to travel answers and must never reach an LLM prompt.
EXCLUDED_COLUMNS = ("Traveler name", "Traveler gender")


@dataclass
class Document:
    """A unit of retrievable text with an ID and filterable metadata.

    Mirrors the shape used by LangChain and Chroma so the pipeline can move
    to a vector store without changing callers.
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_trips(path: Path = TRIPS_CSV_PATH) -> pd.DataFrame:
    """Read the cleaned trips CSV, drop excluded columns, and parse dates."""
    trips = pd.read_csv(path).drop(columns=list(EXCLUDED_COLUMNS))
    trips["Start date"] = pd.to_datetime(trips["Start date"])
    return trips


def format_location(trip: pd.Series) -> str:
    """Return 'City, Country', or only the country when no city is recorded."""
    if pd.isna(trip["City"]):
        return trip["Country"]
    return f"{trip['City']}, {trip['Country']}"


def trip_to_document(trip: pd.Series) -> Document:
    """Convert one trip row into a Document, describing only recorded fields."""
    month = trip["Start date"].strftime("%B")

    sentences = [
        f"{trip['Duration (days)']}-day trip to {format_location(trip)} starting in {month}.",
        f"Accommodation: {trip['Accommodation type']} (${trip['Accommodation cost']:,.0f}).",
    ]
    if pd.isna(trip["Transportation type"]):
        sentences.append("Transportation: not recorded.")
    else:
        sentences.append(
            f"Transportation: {trip['Transportation type']} (${trip['Transportation cost']:,.0f})."
        )
    if pd.notna(trip["Total cost"]):
        sentences.append(f"Total cost: ${trip['Total cost']:,.0f}.")
    sentences.append(f"Traveler: {trip['Traveler age']}-year-old {trip['Traveler nationality']}.")

    metadata = {
        "country": trip["Country"],
        "month": month,
        "duration_days": int(trip["Duration (days)"]),
        "accommodation_type": trip["Accommodation type"],
    }
    optional_fields = {
        "city": trip["City"],
        "transportation_type": trip["Transportation type"],
        "total_cost_usd": float(trip["Total cost"]),
    }
    # Vector stores reject null metadata values, so missing fields are omitted.
    metadata.update({key: value for key, value in optional_fields.items() if pd.notna(value)})

    return Document(id=f"trip-{trip['Trip ID']}", text=" ".join(sentences), metadata=metadata)


def load_documents(path: Path = TRIPS_CSV_PATH) -> list[Document]:
    """Load all trips from the CSV as Documents."""
    trips = load_trips(path)
    return [trip_to_document(trip) for _, trip in trips.iterrows()]


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.\n")
    for document in documents[:2]:
        print(document.id)
        print(document.text)
        print(document.metadata, "\n")