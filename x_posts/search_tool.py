"""
firecrawl_tools.py

LLM-compatible tool definitions wrapping FirecrawlLLMTools.
Provides:
  - TOOL_DEFINITIONS  : list of Anthropic-style tool schemas to pass to the API
  - execute_tool()    : dispatcher — call this with (tool_name, tool_input) from
                        the model's tool_use block

Usage:
    from firecrawl_tools import TOOL_DEFINITIONS, execute_tool

    # Pass TOOL_DEFINITIONS to the LLM
    response = client.messages.create(
        model="...",
        tools=TOOL_DEFINITIONS,
        messages=[...],
    )

    # When the model returns a tool_use block, dispatch it:
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            # result is always a plain dict — feed it back as tool_result
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from integrations.firecrawl import FirecrawlTool
from integrations.firecrawl_utils import FirecrawlLLMTools


# ------------------------------------------------------------------ #
#  Singleton — shared across all tool calls in a session             #
# ------------------------------------------------------------------ #

_tools_instance: Optional[FirecrawlLLMTools] = None


def _get_tools() -> FirecrawlLLMTools:
    global _tools_instance
    if _tools_instance is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY env var is not set.")
        fc = FirecrawlTool(api_key=api_key)
        _tools_instance = FirecrawlLLMTools(
            fc,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _tools_instance


# ------------------------------------------------------------------ #
#  Tool schemas (Anthropic tool-use format)                          #
# ------------------------------------------------------------------ #

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for a query and return the top results. "
            "Optionally fetches and cleans the full page content for each result. "
            "Can optionally run an LLM filter over each result to extract a structured "
            "summary, key points, and relevant content — set llm_filter=true for richer output. "
            "Use this when you need to find information across multiple sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                    "default": 5,
                },
                "include_content": {
                    "type": "boolean",
                    "description": (
                        "If true, fetches and returns cleaned markdown content for each result. "
                        "Slower but more informative. Default false."
                    ),
                    "default": False,
                },
                "max_chars_per_result": {
                    "type": "integer",
                    "description": "Max characters of raw content to return per result. Default 10000.",
                    "default": 10000,
                },
                "llm_filter": {
                    "type": "boolean",
                    "description": (
                        "If true, runs each result's content through an LLM to produce "
                        "structured filtered_content (title, summary, content, key_points). "
                        "Requires include_content=true. Slower but much cleaner output. Default false."
                    ),
                    "default": False,
                },
                "only_main_content": {
                    "type": "boolean",
                    "description": "Strip nav/footer server-side. Default true.",
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Scrape a single URL and return its cleaned content. "
            "Automatically handles sites that block headless browsers (e.g. Reddit) "
            "by falling back to native API fetchers. "
            "Set llm_filter=true to get a structured extraction (title, summary, key_points, content) "
            "instead of raw markdown. Use this when you have a specific URL to read."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full HTTP/HTTPS URL to fetch.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "What you are looking for on this page. Used as context when "
                        "llm_filter=true to guide content extraction. Optional."
                    ),
                    "default": "",
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "html", "text"],
                    "description": "Output format for the scraped content. Default 'markdown'.",
                    "default": "markdown",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of content to return. Default 10000.",
                    "default": 10000,
                },
                "only_main_content": {
                    "type": "boolean",
                    "description": "Strip nav/footer server-side. Default true.",
                    "default": True,
                },
                "llm_filter": {
                    "type": "boolean",
                    "description": (
                        "If true, runs OpenAI over the raw content and returns "
                        "structured filtered_content (title, summary, content, key_points) "
                        "alongside the clipped raw content. Default false."
                    ),
                    "default": False,
                },
            },
            "required": ["url"],
        },
    },
]


# ------------------------------------------------------------------ #
#  Individual tool handlers                                           #
# ------------------------------------------------------------------ #

def _handle_web_search(inp: Dict[str, Any]) -> Dict[str, Any]:
    query: str = inp.get("query", "")
    k: int = int(inp.get("k", 5))
    include_content: bool = bool(inp.get("include_content", False))
    max_chars_per_result: int = int(inp.get("max_chars_per_result", 10_000))
    llm_filter: bool = bool(inp.get("llm_filter", False))
    only_main_content: bool = bool(inp.get("only_main_content", True))

    return _get_tools().web_search(
        query,
        k=k,
        include_content=include_content,
        max_chars_per_result=max_chars_per_result,
        only_main_content=only_main_content,
        llm_filter=llm_filter,
    )


def _handle_web_fetch(inp: Dict[str, Any]) -> Dict[str, Any]:
    url: str = inp.get("url", "")
    query: str = inp.get("query", "")
    fmt: str = inp.get("format", "markdown")
    max_chars: int = int(inp.get("max_chars", 10_000))
    only_main_content: bool = bool(inp.get("only_main_content", True))
    llm_filter: bool = bool(inp.get("llm_filter", False))

    return _get_tools().web_fetch(
        url,
        query=query,
        format=fmt,
        max_chars=max_chars,
        only_main_content=only_main_content,
        llm_filter=llm_filter,
    )


# ------------------------------------------------------------------ #
#  Public dispatcher                                                  #
# ------------------------------------------------------------------ #

_HANDLERS = {
    "web_search": _handle_web_search,
    "web_fetch":  _handle_web_fetch,
}


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a tool_use block from the LLM.

    Args:
        tool_name:  The `name` field from the model's tool_use block.
        tool_input: The `input` dict from the model's tool_use block.

    Returns:
        A plain dict (already JSON-serialisable) that can be sent back
        as a tool_result content block.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return {
            "ok": False,
            "data": None,
            "error": {
                "type": "UnknownTool",
                "message": f"No handler for tool '{tool_name}'. Available: {list(_HANDLERS)}",
            },
        }

    try:
        return handler(tool_input)
    except Exception as exc:
        return {
            "ok": False,
            "data": None,
            "error": {
                "type": "ToolException",
                "message": str(exc),
            },
        }


# ------------------------------------------------------------------ #
#  Quick smoke-test (python firecrawl_tools.py)                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== web_search smoke test ===\n")
    out = execute_tool(
        "web_search",
        {
            "query": "LiveKit Python SDK real-time audio track",
            "k": 2,
            "include_content": True,
            "llm_filter": True,
        },
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n=== web_fetch smoke test (Reddit) ===\n")
    out2 = execute_tool(
        "web_fetch",
        {
            "url": "https://medium.com/@mridulrao674385/ai-agents-and-observability-the-environment-regime-problem-86b41f16b0e4",
            "query": "",
            "llm_filter": True,
        },
    )
    print(json.dumps(out2, indent=2, ensure_ascii=False))