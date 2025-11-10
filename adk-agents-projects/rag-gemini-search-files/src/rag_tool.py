"""RAG querying helpers around Gemini File Search.

This module exposes a simple function to query a File Search Store as a tool
for ADK agents or direct CLI usage. It also provides small helpers to build
the proper File Search Tool configuration.
"""

from __future__ import annotations

from typing import Any, List, Dict
from dataclasses import asdict, is_dataclass
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


def format_citations_inline(sources: list[dict[str, Any]], cfg: "RagConfig") -> str:
    """Return an inline sources string according to formatting config.

    Styles:
      - parentheses: "Fontes: (Doc1, Art 3; Doc2)"
      - brackets:    "Sources: [1:Doc1] [2:Doc2]"
    """
    if not sources:
        return ""

    label = (getattr(cfg.formatting, "sources_label", None) or "Sources:").strip()
    style = getattr(cfg.formatting, "style", "parentheses")
    joiner = getattr(cfg.formatting, "joiner", "; ")
    show_page = bool(getattr(cfg.formatting, "show_page", True))
    page_prefix = getattr(cfg.formatting, "page_prefix", "p.").strip()

    if style == "brackets":
        items = []
        for i, s in enumerate(sources, 1):
            name = s.get("file_name") or s.get("file_resource") or "source"
            items.append(f"[{i}:{str(name).replace(' ', '_')}]")
        return f"{label} {' '.join(items)}".strip()

    # Default: parentheses list with optional page
    items = []
    for s in sources:
        name = s.get("file_name") or s.get("file_resource") or "source"
        part = str(name)
        if show_page and (s.get("page") is not None):
            part = f"{part}, {page_prefix} {s['page']}"
        items.append(part)
    return f"{label} (" + joiner.join(items) + ")"


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

    # Extract human-friendly citations: file display name and optional chunk/page info.
    sources: List[Dict[str, Any]] = []
    raw_grounding: Dict[str, Any] | None = None
    try:
        cand0 = resp.candidates[0]
        grounding = getattr(cand0, "grounding_metadata", None) or {}
        # Best-effort serialization for debug
        try:
            if is_dataclass(grounding):
                raw_grounding = asdict(grounding)
            else:
                raw_grounding = grounding.__dict__ if hasattr(grounding, "__dict__") else dict(grounding)
        except Exception:
            raw_grounding = None
        # Newer SDKs may expose 'citations' directly under grounding metadata
        for c in getattr(grounding, "citations", []) or []:
            src = getattr(c, "source", None)
            if src and getattr(src, "file", None):
                f = src.file
                sources.append({
                    "type": "file",
                    "file_name": getattr(f, "display_name", None) or getattr(f, "name", None),
                    "file_resource": getattr(f, "name", None),
                    "page": getattr(src, "page", None),
                    "chunk": getattr(src, "chunk", None),
                })
        # Fallbacks: sometimes only 'supporting_references' or 'grounding_chunks' exist
        if not sources:
            refs = getattr(grounding, "supporting_references", []) or []
            for r in refs:
                f = getattr(r, "file", None)
                if f:
                    sources.append({
                        "type": "file",
                        "file_name": getattr(f, "display_name", None) or getattr(f, "name", None),
                        "file_resource": getattr(f, "name", None),
                        "page": getattr(r, "page", None),
                        "chunk": getattr(r, "chunk", None),
                    })
        # Last resort: inspect grounding chunks if exposed
        if not sources:
            chunks = getattr(grounding, "grounding_chunks", []) or []
            for ch in chunks:
                f = getattr(ch, "file", None)
                if f:
                    sources.append({
                        "type": "file",
                        "file_name": getattr(f, "display_name", None) or getattr(f, "name", None),
                        "file_resource": getattr(f, "name", None),
                        "page": getattr(ch, "page", None),
                        "chunk": getattr(ch, "chunk", None),
                    })
    except Exception:
        pass

    return {
        "status": "success",
        "model": model_name,
        "answer": text,
        "citations": sources if sources else None,
        "raw_grounding": raw_grounding,
    }
