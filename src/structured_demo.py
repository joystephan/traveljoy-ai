# structured_demo.py — TravelJoy AI: structured outputs demo (Gemini free tier)
import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel

load_dotenv()  # reads GEMINI_API_KEY + model names from .env


# The "shape" we want the answer to have
class TripRecommendation(BaseModel):
    city: str
    country: str
    best_month: str
    estimated_budget_usd: int                              # real int, not "around $1,200"
    travel_style: Literal["budget", "mid-range", "luxury"]  # only these 3 allowed
    reasons: list[str]


client = genai.Client()  # finds GEMINI_API_KEY automatically

# Model names come from .env, so one change updates the whole project.
# The "if m" skips any model name missing from .env.
MODELS = [m for m in (os.getenv("GEMINI_MODEL"), os.getenv("GEMINI_FALLBACK_MODEL")) if m]


def ask(prompt: str) -> TripRecommendation:
    """Send a prompt, get back a validated TripRecommendation.
    Tries the main model first, then falls back if it's busy or over quota."""
    if not MODELS:
        raise RuntimeError("No models set. Add GEMINI_MODEL to your .env file.")

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",  # force JSON, not prose
                    "response_schema": TripRecommendation,     # force OUR shape
                },
            )
            print(f"(answered by {model})")
            return response.parsed  # already validated into our class
        except errors.APIError as e:
            if e.code in (429, 503):  # quota hit or server busy: try the next model
                print(f"{model} unavailable ({e.code}), trying fallback...")
                continue
            raise  # any other error is a real bug, so show it
    raise RuntimeError("All models unavailable, try again later.")


# Runs only when you execute this file directly, not when another file imports ask()
if __name__ == "__main__":
    trip = ask("Recommend one budget-friendly European city to visit in March for a 5-day trip.")
    print(trip)
    print(f"\n{trip.city}, {trip.country}: ~${trip.estimated_budget_usd} ({trip.travel_style})")
    for r in trip.reasons:
        print(" -", r)