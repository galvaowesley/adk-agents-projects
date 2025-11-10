"""Configuration loading utilities for the RAG Gemini File Search system.

This module defines dataclasses representing distinct configuration domains
(chunking, retrieval, model, store, prompt) and provides helpers to load
their values from YAML files. Missing files gracefully fall back to
dataclass defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.getenv("RAG_CONFIG_DIR", BASE_DIR / "configs"))


@dataclass
class ChunkingConfig:
    """Chunking parameters used when importing documents.

    Args:
        max_tokens_per_chunk: Maximum tokens per chunk produced by white space chunker.
        max_overlap_tokens: Overlapping tokens between consecutive chunks.
    """

    max_tokens_per_chunk: int = 200
    max_overlap_tokens: int = 20


@dataclass
class RetrievalConfig:
    """Retrieval parameters for semantic search.

    Args:
        top_k: Maximum number of results to request per query.
        metadata_filter: Optional metadata filter expression to narrow documents.
    """

    top_k: int = 5
    metadata_filter: str | None = None


@dataclass
class ModelConfig:
    """Generation model parameters.

    Args:
        name: Model identifier.
        temperature: Sampling temperature for generation.
        max_output_tokens: Maximum number of tokens to produce.
    """

    name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 1024


@dataclass
class StoreConfig:
    """File Search store parameters.

    Args:
        display_name: Human friendly display name used to locate or create store.
        create_if_missing: Whether creation is performed automatically if not found.
        polling_interval_seconds: Interval between status checks for import operations.
        timeout_seconds: Maximum time to await completion of a long running import.
    """

    display_name: str = "rag-store-principal"
    create_if_missing: bool = True
    polling_interval_seconds: int = 5
    timeout_seconds: int = 180


@dataclass
class FormattingConfig:
    """Answer formatting parameters.

    Args:
        append_sources_inline: Whether to append sources at the end of the answer text.
        sources_label: Label prefix for sources section (e.g., "Sources:" or "Fontes:").
        style: Formatting style for inline sources. Options: "parentheses", "brackets".
        joiner: Separator to use between items.
        show_page: Whether to include page information when available.
        page_prefix: Prefix used before the page number (e.g., "p.", "Art").
    """

    append_sources_inline: bool = True
    sources_label: str = "Sources:"
    style: str = "parentheses"  # or "brackets"
    joiner: str = "; "
    show_page: bool = True
    page_prefix: str = "p."


@dataclass
class RagConfig:
    """Aggregate configuration combining all sub-config sections.

    Args:
        chunking: Chunking configuration.
        retrieval: Retrieval/search parameters.
        model: Generation model parameters.
        store: File search store parameters.
    """

    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    model: ModelConfig
    store: StoreConfig
    formatting: FormattingConfig


@dataclass
class PromptConfig:
    """Prompt configuration holding a system instruction.

    Args:
        system_prompt: System level instruction injected during generation.
    """

    system_prompt: str


def load_yaml(path: Path) -> dict[str, object]:
    """Load a YAML file returning an empty dict when missing or empty.

    Args:
        path: Path object pointing to a YAML file.

    Returns:
        dict[str, object]: Parsed YAML content or empty dict if file missing/empty.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def load_rag_config() -> RagConfig:
    """Load composite RAG configuration from YAML files.

    Returns:
        RagConfig: Fully populated configuration with defaults applied.
    """
    cfg = load_yaml(CONFIG_DIR / "rag_config.yml")
    chunking = ChunkingConfig(**cfg.get("chunking", {}))
    retrieval = RetrievalConfig(**cfg.get("retrieval", {}))
    model = ModelConfig(**cfg.get("model", {}))
    store = StoreConfig(**cfg.get("store", {}))
    formatting = FormattingConfig(**cfg.get("formatting", {}))
    return RagConfig(
        chunking=chunking,
        retrieval=retrieval,
        model=model,
        store=store,
        formatting=formatting,
    )


def load_prompt_config() -> PromptConfig:
    """Load the system prompt configuration.

    Returns:
        PromptConfig: Prompt dataclass with system instruction.
    """
    data = load_yaml(CONFIG_DIR / "system_prompt.yml")
    return PromptConfig(system_prompt=data.get("system_prompt", ""))
