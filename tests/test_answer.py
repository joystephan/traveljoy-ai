"""Tests for src.rag.answer: prompt building, the no-match guard, and model call wiring.

The Gemini call is replaced with a fake, so these tests never reach the real API.
"""

import pytest

from src.rag import answer
from src.rag.answer import (
    NO_MATCH_MESSAGE,
    SYSTEM_INSTRUCTION,
    Answer,
    answer_question,
    build_prompt,
)
from src.rag.ingest import Document
from src.rag.retrieve import DEFAULT_TOP_K

MATCHING_QUESTION = "hotel trip to London in May"
FAKE_REPLY = "  Trip-1 fits best [trip-1].  \n"


@pytest.fixture(autouse=True)
def llm_calls(monkeypatch):
    """Replace generate_text with a fake that records each call and returns FAKE_REPLY.

    Autouse, so no test in this module can reach the real API by accident.
    """
    calls = []

    def fake_generate_text(prompt, system_instruction=None):
        calls.append({"prompt": prompt, "system_instruction": system_instruction})
        return FAKE_REPLY

    # answer.py imports generate_text by name, so it must be replaced in that module.
    monkeypatch.setattr(answer, "generate_text", fake_generate_text)
    return calls


def test_build_prompt_tags_each_trip_with_its_id():
    documents = [Document(id="trip-1", text="Trip A."), Document(id="trip-2", text="Trip B.")]
    assert build_prompt("Which is cheaper?", documents) == (
        "Trips:\n[trip-1] Trip A.\n[trip-2] Trip B.\n\nQuestion: Which is cheaper?"
    )


def test_no_matching_trips_skips_the_model(llm_calls):
    assert answer_question("submarine") == Answer(text=NO_MATCH_MESSAGE)
    assert llm_calls == []


def test_matching_trips_are_sent_with_system_instruction(llm_calls):
    answer_question(MATCHING_QUESTION)
    assert len(llm_calls) == 1
    call = llm_calls[0]
    assert call["system_instruction"] == SYSTEM_INSTRUCTION
    assert "[trip-1]" in call["prompt"]
    assert call["prompt"].endswith(f"Question: {MATCHING_QUESTION}")


def test_answer_is_stripped_and_lists_retrieved_trips():
    result = answer_question(MATCHING_QUESTION)
    assert result.text == FAKE_REPLY.strip()
    assert result.source_ids[0] == "trip-1"
    assert len(result.source_ids) == DEFAULT_TOP_K


def test_top_k_limits_trips_sent_to_the_model(llm_calls):
    result = answer_question(MATCHING_QUESTION, top_k=2)
    assert len(result.source_ids) == 2
    assert llm_calls[0]["prompt"].count("[trip-") == 2


def test_model_errors_reach_the_caller(monkeypatch):
    def unavailable_model(prompt, system_instruction=None):
        raise RuntimeError("All Gemini models are unavailable.")

    # main.py turns this error into a friendly message, so answer_question must not swallow it.
    monkeypatch.setattr(answer, "generate_text", unavailable_model)
    with pytest.raises(RuntimeError):
        answer_question(MATCHING_QUESTION)