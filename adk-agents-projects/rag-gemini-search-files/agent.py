"""ADK agent wired with a RAG file search tool.

This module exposes a root Agent that can leverage Gemini File Search via a
tool function. Configuration (model and system prompt) is loaded from YAML
using helpers in the package.

CLI usage (from project root):
    python agent.py "Question here"
"""

from pathlib import Path
import sys
import importlib
from google.adk.agents.llm_agent import Agent

# Ensure we can import modules from ./src when running from project root
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    rag_tool = importlib.import_module("rag_tool")
    config_mod = importlib.import_module("config")
except ModuleNotFoundError:
    rag_tool = importlib.import_module("src.rag_tool")
    config_mod = importlib.import_module("src.config")


prompt_cfg = config_mod.load_prompt_config()
rag_cfg = config_mod.load_rag_config()

def file_search_tool(query: str) -> dict:
    """ADK tool that queries Gemini File Search (RAG).

    Args:
        query: User natural language question.

    Returns:
        dict: A dictionary with keys "status", "model", "answer" and optional
            "citations" (grounding metadata) when available.
    """
    result = rag_tool.rag_query(query)
    cites = result.get("citations") or []
    if cites and getattr(rag_cfg, "formatting", None) and rag_cfg.formatting.append_sources_inline:
        inline = rag_tool.format_citations_inline(cites, rag_cfg)
        # Single trailing newline keeps formatting consistent for ADK UI.
        result["answer"] = f"{result.get('answer','')}\n{inline}".rstrip()
    return result


root_agent = Agent(
    model=rag_cfg.model.name,
    name="rag_root_agent",
    description="RAG agent that queries documents via Gemini File Search.",
    instruction=prompt_cfg.system_prompt,
    tools=[file_search_tool],
)


def main() -> None:
    """Simple CLI entrypoint that calls the registered RAG tool.

    Note: This executes the tool directly for convenience. The constructed
    `root_agent` is available for richer ADK flows if needed.
    """
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarize the key topics from the documents."
    result = file_search_tool(question)
    answer = result.get("answer", "") or ""
    print(answer)
    # Print citations if available
    cites = result.get("citations") or []
    if cites:
        print("\nSources:")
        for i, c in enumerate(cites, 1):
            label = c.get("file_name") or c.get("file_resource") or "unknown"
            details = []
            if c.get("page") is not None:
                details.append(f"page {c['page']}")
            if c.get("chunk") is not None:
                details.append(f"chunk {c['chunk']}")
            suffix = f" ({', '.join(details)})" if details else ""
            print(f"  [{i}] {label}{suffix}")
    else:
        # Fallback: attempt to derive file names from raw grounding if present
        raw_grounding = result.get("raw_grounding") or {}
        files = []
        try:
            # Common keys that might hold file references
            for key in ["citations", "supporting_references", "grounding_chunks"]:
                for entry in raw_grounding.get(key, []) or []:
                    f = getattr(entry, "file", None) if hasattr(entry, "file") else entry.get("file") if isinstance(entry, dict) else None
                    if f:
                        fname = getattr(f, "display_name", None) or getattr(f, "name", None) or (
                            f.get("display_name") if isinstance(f, dict) else None
                        ) or (f.get("name") if isinstance(f, dict) else None)
                        if fname:
                            files.append(fname)
            unique_files = list(dict.fromkeys(files))
            if unique_files:
                print("\nSources (fallback):")
                for i, name in enumerate(unique_files, 1):
                    print(f"  [{i}] {name}")
        except Exception:
            pass


if __name__ == "__main__":
    main()