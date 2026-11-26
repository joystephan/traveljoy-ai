"""Command-line interface for asking TravelJoy AI travel questions.

Run from the project root:
    python main.py

Type a question to get an answer grounded in the trip dataset, along with
the IDs of the trips retrieved for it. Type "quit" or "exit", submit an empty
line, or press Ctrl+C to leave.
"""

import logging

from src.rag.answer import answer_question

EXIT_COMMANDS = {"quit", "exit"}
INPUT_PROMPT = "\nAsk a travel question (or 'quit'): "
UNAVAILABLE_MESSAGE = (
    "Sorry, the AI service is unavailable right now. Please try again in a moment."
)


def configure_logging() -> None:
    """Show warnings from all libraries and model-fallback details from src.llm."""
    # INFO is enabled for our own logger only: third-party loggers at INFO/DEBUG
    # can print request details that should never appear in the terminal.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("src.llm").setLevel(logging.INFO)


def run() -> None:
    """Read questions in a loop and print grounded answers until the user exits."""
    print("TravelJoy AI — ask about trips in the dataset.")

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
            answer = answer_question(question)
        except RuntimeError:
            # Raised by src.llm when every configured model failed.
            print(UNAVAILABLE_MESSAGE)
            continue

        print(f"\n{answer.text}")
        if answer.source_ids:
            print(f"\nRetrieved trips: {', '.join(answer.source_ids)}")

    print("Goodbye!")


if __name__ == "__main__":
    configure_logging()
    try:
        run()
    except KeyboardInterrupt:
        # Ctrl+C can arrive while waiting for input or mid-request.
        print("\nGoodbye!")
