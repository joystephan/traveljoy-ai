"""Keyword retrieval over trip documents.

Ranks documents by the query terms they share, weighting rare terms
(e.g. a city name) above common ones (e.g. "trip") using inverse document
frequency. This is a baseline for embedding search with a vector store;
the search() interface is designed to stay the same.

Usage:
    python -m src.rag.retrieve "hotel trip to London in May"
"""

import math
import re
import sys
from collections import Counter

from src.rag.ingest import Document, load_documents

DEFAULT_TOP_K = 5
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Return the set of lowercase word tokens in the text."""
    return set(TOKEN_PATTERN.findall(text.lower()))


class KeywordRetriever:
    """Rank documents by shared query terms, weighted by inverse document frequency."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._doc_tokens = [tokenize(doc.text) for doc in documents]
        doc_frequency = Counter(
            token for tokens in self._doc_tokens for token in tokens
        )
        # Terms found in every document (e.g. "trip", "to") get zero weight,
        # so no stop-word list is needed.
        self._idf = {
            token: math.log(len(documents) / count)
            for token, count in doc_frequency.items()
        }

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Document]:
        """Return up to top_k documents most relevant to the query, best first.

        Documents that share no weighted terms with the query are not returned.
        """
        query_tokens = tokenize(query)
        scored = []
        for document, tokens in zip(self.documents, self._doc_tokens):
            score = sum(self._idf[token] for token in query_tokens & tokens)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [document for _, document in scored[:top_k]]


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "hotel trip to London in May"
    retriever = KeywordRetriever(load_documents())
    results = retriever.search(query)
    print(f"Query: {query}")
    print(f"Found {len(results)} matching trips.\n")
    for document in results:
        print(f"{document.id}: {document.text}")
