"""Tests for src.rag.retrieve: tokenizing, IDF ranking, and the search() contract."""

import pytest

from src.rag.ingest import Document, load_documents
from src.rag.retrieve import DEFAULT_TOP_K, KeywordRetriever, tokenize


@pytest.fixture
def small_retriever():
    # Small enough to compute IDF by hand: "london" is in 1 of 3 documents,
    # "may" in 2 of 3, and "trip" in all 3.
    documents = [
        Document(id="london-may", text="Hotel trip to London in May."),
        Document(id="paris-may", text="Hostel trip to Paris in May."),
        Document(id="bali-july", text="Resort trip to Bali in July."),
    ]
    return KeywordRetriever(documents)


@pytest.fixture(scope="module")
def trip_retriever():
    return KeywordRetriever(load_documents())


def result_ids(results):
    return [document.id for document in results]


def test_tokenize_lowercases_strips_punctuation_and_deduplicates():
    assert tokenize("London, london! 7-day") == {"london", "7", "day"}


def test_rare_terms_outrank_common_terms(small_retriever):
    assert result_ids(small_retriever.search("London May")) == ["london-may", "paris-may"]


def test_search_is_case_insensitive(small_retriever):
    assert small_retriever.search("LONDON") == small_retriever.search("london")


def test_term_in_every_document_matches_nothing(small_retriever):
    assert small_retriever.search("trip") == []


def test_query_with_no_shared_terms_returns_empty(small_retriever):
    assert small_retriever.search("submarine") == []


def test_empty_query_returns_empty(small_retriever):
    assert small_retriever.search("") == []


def test_search_respects_top_k(small_retriever):
    assert len(small_retriever.search("May", top_k=1)) == 1


def test_default_top_k_limits_real_results(trip_retriever):
    # The dataset has more than DEFAULT_TOP_K London trips.
    assert len(trip_retriever.search("London")) == DEFAULT_TOP_K


def test_best_match_ranks_first_on_real_trips(trip_retriever):
    assert trip_retriever.search("hotel trip to London in May")[0].id == "trip-1"