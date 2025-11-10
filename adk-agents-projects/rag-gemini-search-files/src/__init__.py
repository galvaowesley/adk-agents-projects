"""RAG package integrating Gemini File Search utilities.

Exports:
	rag_query: High-level helper to perform a retrieval-augmented query.
"""

from rag_tool import rag_query  # re-export helper

__all__ = ["rag_query"]
__version__ = "0.1.0"
