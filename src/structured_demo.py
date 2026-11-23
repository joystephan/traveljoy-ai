"""Structured output demo for TravelJoy AI.

Requests a destination recommendation from Gemini and validates the
response against a Pydantic schema.

Usage:
    python -m src.structured_demo
"""
import logging
from typing import Literal

from pydantic import BaseModel

from src.llm import generate_structured


class TripRecommendation(BaseModel):
    """A single destination recommendation returned by the model."""

    city: str
    country: str
    best_month: str
    estimated_budget_usd: int
    travel_style: Literal["budget", "mid-range", "luxury"]
    reasons: list[str]


def recommend(prompt: str) -> TripRecommendation:
    """Return a validated TripRecommendation for the given prompt."""
    return generate_structured(prompt, TripRecommendation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("src.llm").setLevel(logging.INFO)

    trip = recommend("Recommend one budget-friendly European city to visit in March for a 5-day trip.")
    print(trip)
    print(f"\n{trip.city}, {trip.country}: ~${trip.estimated_budget_usd} ({trip.travel_style})")
    for reason in trip.reasons:
        print(" -", reason)