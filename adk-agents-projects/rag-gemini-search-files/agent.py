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
    return rag_tool.rag_query(query)


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
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()