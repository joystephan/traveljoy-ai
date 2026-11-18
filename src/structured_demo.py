"""Structured output demo for TravelJoy AI.

Requests a destination recommendation from Gemini and validates the
response against a Pydantic schema. Falls back to a secondary model
on transient API errors (rate limits, overload).
"""
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel

load_dotenv()

TRANSIENT_ERROR_CODES = (429, 503)


class TripRecommendation(BaseModel):
    city: str
    country: str
    best_month: str
    estimated_budget_usd: int
    travel_style: Literal["budget", "mid-range", "luxury"]
    reasons: list[str]


client = genai.Client()

# Model names are configured in .env so they can change without code edits.
MODELS = [m for m in (os.getenv("GEMINI_MODEL"), os.getenv("GEMINI_FALLBACK_MODEL")) if m]


def ask(prompt: str) -> TripRecommendation:
    """Return a validated TripRecommendation for the given prompt.

    Tries each configured model in order. Transient errors move on to the
    next model; any other API error is raised immediately.
    """
    if not MODELS:
        raise RuntimeError("No models configured. Set GEMINI_MODEL in .env.")

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": TripRecommendation,
                },
            )
            print(f"(answered by {model})")
            return response.parsed
        except errors.APIError as e:
            if e.code in TRANSIENT_ERROR_CODES:
                print(f"{model} unavailable ({e.code}), trying fallback...")
                continue
            raise
    raise RuntimeError("All configured models are unavailable. Try again later.")


if __name__ == "__main__":
    trip = ask("Recommend one budget-friendly European city to visit in March for a 5-day trip.")
    print(trip)
    print(f"\n{trip.city}, {trip.country}: ~${trip.estimated_budget_usd} ({trip.travel_style})")
    for reason in trip.reasons:
        print(" -", reason)