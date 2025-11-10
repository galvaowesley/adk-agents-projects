"""RAG querying helpers around Gemini File Search.

This module exposes a simple function to query a File Search Store as a tool
for ADK agents or direct CLI usage. It also provides small helpers to build
the proper File Search Tool configuration.
"""

from __future__ import annotations

from typing import Any
import os

from google import genai
from google.genai import types
from rich.console import Console

from config import RagConfig, load_prompt_config, load_rag_config
from ingestion import ensure_store

console = Console()


def _client() -> genai.Client:
    """Create a Gemini client using only the ``GOOGLE_API_KEY`` environment variable.

    Returns:
        genai.Client: Authenticated client.
    Raises:
        RuntimeError: If the variable is not defined.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Export it before querying.")
    return genai.Client(api_key=api_key)


def _file_search_tool(config: RagConfig, client: genai.Client) -> types.Tool:
    """Build a File Search tool bound to the configured store.

    Args:
        config: RAG configuration.
        client: Gemini API client.

    Returns:
        types.Tool: Tool instance referencing the resolved store.
    """
    store = ensure_store(client, config.store.display_name)
    return types.Tool(file_search=types.FileSearch(file_search_store_names=[store.name]))


def rag_query(question: str, top_k: int | None = None, metadata_filter: str | None = None) -> dict[str, Any]:
    """Query the File Search Store via Gemini API and return answer plus citations.

    Note: ``top_k`` and ``metadata_filter`` are currently ignored because the
    SDK's ``FileSearch`` model does not accept those fields (would raise
    validation errors). They are kept as parameters for future compatibility.

    Args:
        question: Natural language question.
        top_k: Placeholder for future override of number of retrieved docs.
        metadata_filter: Placeholder for future metadata filtering.

    Returns:
        dict[str, Any]: A dictionary with keys:
            - status: Operation status string ("success" if OK).
            - model: Model identifier used.
            - answer: Text answer produced by the model.
            - citations: Grounding metadata when available, else None.
    """
    cfg = load_rag_config()
    prompt_cfg = load_prompt_config()

    client = _client()

    # FileSearch currently accepts only the store names; other fields like
    # max_num_results or metadata_filter are not supported here by the SDK
    # model (pydantic validation fails). Retrieval behavior is controlled by
    # the model and store configuration.
    store_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[ensure_store(client, cfg.store.display_name).name]
        )
    )

    gen_cfg = types.GenerateContentConfig(
        system_instruction=prompt_cfg.system_prompt,
        tools=[store_tool],
    )

    model_name = cfg.model.name

    resp = client.models.generate_content(
        model=model_name,
        contents=question,
        config=gen_cfg,
    )

    text = getattr(resp, "text", None)
    citations = None
    try:
        cand0 = resp.candidates[0]
        grounding = getattr(cand0, "grounding_metadata", None)
        citations = grounding
    except Exception:
        citations = None

    return {"status": "success", "model": model_name, "answer": text, "citations": citations}
