"""Tests for src.rag.ingest: privacy filtering, missing-data handling, and age groups."""

import re

import pandas as pd
import pytest

from src.rag.ingest import TRIPS_CSV_PATH, format_age_group, load_documents, load_trips

EXPECTED_TRIP_COUNT = 137

# Defined independently of ingest.EXCLUDED_COLUMNS so the test fails if that constant is weakened.
IDENTITY_COLUMNS = {"Traveler name", "Traveler gender", "Traveler nationality"}


@pytest.fixture(scope="module")
def documents():
    return load_documents()


@pytest.fixture(scope="module")
def documents_by_id(documents):
    return {document.id: document for document in documents}


@pytest.fixture(scope="module")
def raw_trips():
    # Read the CSV directly: load_trips() already drops the identity columns under test.
    return pd.read_csv(TRIPS_CSV_PATH)


def find_leaks(documents, values):
    """Return (document id, value) pairs where a value appears as a whole word in document text."""
    leaks = []
    for value in values:
        pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
        leaks.extend((document.id, value) for document in documents if pattern.search(document.text))
    return leaks


def test_one_document_per_trip(documents):
    assert len(documents) == EXPECTED_TRIP_COUNT
    assert len({document.id for document in documents}) == EXPECTED_TRIP_COUNT


def test_document_text_contains_no_nan(documents):
    nan_pattern = re.compile(r"\bnan\b", re.IGNORECASE)
    assert [document.id for document in documents if nan_pattern.search(document.text)] == []


def test_metadata_contains_no_missing_values(documents):
    missing = [
        (document.id, key)
        for document in documents
        for key, value in document.metadata.items()
        if pd.isna(value)
    ]
    assert missing == []


def test_load_trips_drops_identity_columns():
    assert IDENTITY_COLUMNS.isdisjoint(load_trips().columns)


def test_no_traveler_names_in_documents(documents, raw_trips):
    names = raw_trips["Traveler name"].dropna().unique()
    assert find_leaks(documents, names) == []


def test_no_traveler_nationalities_in_documents(documents, raw_trips):
    nationalities = raw_trips["Traveler nationality"].dropna().unique()
    assert find_leaks(documents, nationalities) == []


@pytest.mark.parametrize(
    ("age", "expected"),
    [(35, "30s"), (30, "30s"), (39, "30s"), (20, "20s"), (29.9, "20s")],
)
def test_format_age_group(age, expected):
    assert format_age_group(age) == expected


def test_missing_transport_is_described_not_guessed(documents_by_id):
    trip = documents_by_id["trip-83"]
    assert "Transportation: not recorded." in trip.text
    assert "transportation_type" not in trip.metadata
    assert "total_cost_usd" not in trip.metadata


def test_country_only_destination_has_no_city(documents_by_id):
    assert "city" not in documents_by_id["trip-28"].metadata