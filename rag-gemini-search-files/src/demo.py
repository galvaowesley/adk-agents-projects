"""Simple CLI demo to perform a RAG query using the current configuration.

Usage:
    python -m rag_gemini_search_files.demo "What are the main topics in the documents?"
"""

from __future__ import annotations

import sys
import os

from config import load_prompt_config, load_rag_config
from rag_tool import rag_query


def main() -> None:
    """Entry point for the `rag-ask` console script."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY. Export it before running.")
    _ = load_rag_config()  # ensure config can load
    _ = load_prompt_config()
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarize the key topics from the documents."
    result = rag_query(question)
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
