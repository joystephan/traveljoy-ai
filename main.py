"""Command-line interface for asking TravelJoy AI travel questions.

Run from the project root:
    python main.py
    python main.py --top-k 3

Type a question to get an answer grounded in the trip dataset, along with
the IDs of the trips retrieved for it. Type "quit" or "exit", submit an empty
line, or press Ctrl+C to leave.
"""

import argparse
import logging

from src.rag.answer import answer_question
from src.rag.retrieve import DEFAULT_TOP_K

EXIT_COMMANDS = {"quit", "exit"}
INPUT_PROMPT = "\nAsk a travel question (or 'quit'): "
UNAVAILABLE_MESSAGE = (
    "Sorry, the AI service is unavailable right now. Please try again in a moment."
)
# Keyword retrieval needs words found in the data, so examples name a city,
# country, month, or accommodation type.
EXAMPLE_QUESTIONS = (
    "Which London trip had the lowest total cost?",
    "Recommend a resort trip in Thailand.",
    "What hotel trips started in May?",
)


def positive_int(value: str) -> int:
    """Parse a command-line value as an integer of at least 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args() -> argparse.Namespace:
    """Return the parsed command-line options."""
    parser = argparse.ArgumentParser(description="Ask TravelJoy AI travel questions.")
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help=f"number of trips retrieved per question (default: {DEFAULT_TOP_K})",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Show warnings from all libraries and model-fallback details from src.llm."""
    # INFO is enabled for our own logger only: third-party loggers at INFO/DEBUG
    # can print request details that should never appear in the terminal.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("src.llm").setLevel(logging.INFO)


def print_welcome() -> None:
    """Print a short introduction with example questions."""
    print("TravelJoy AI — ask about trips in the dataset.")
    print("For example:")
    for question in EXAMPLE_QUESTIONS:
        print(f"  - {question}")


def run(top_k: int = DEFAULT_TOP_K) -> None:
    """Read questions in a loop and print grounded answers until the user exits."""
    print_welcome()

    while True:
        try:
            question = input(INPUT_PROMPT).strip()
        except EOFError:
            # Ctrl+D or piped input ending.
            print()
            break

        if not question or question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = answer_question(question, top_k=top_k)
        except RuntimeError:
            # Raised by src.llm when every configured model failed.
            print(UNAVAILABLE_MESSAGE)
            continue

        print(f"\n{answer.text}")
        if answer.source_ids:
            print(f"\nRetrieved trips: {', '.join(answer.source_ids)}")

    print("Goodbye!")


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    try:
        run(top_k=args.top_k)
    except KeyboardInterrupt:
        # Ctrl+C can arrive while waiting for input or mid-request.
        print("\nGoodbye!")
