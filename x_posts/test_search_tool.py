#!/usr/bin/env python3
"""
test_firecrawl_tool.py

Sanity test for FirecrawlTool.

Runs:
  1) lookup_tool() on a specific URL (your Medium article)
  2) search_tool() on a generic query: "what is the current trending crypto"
     - once with include_content=False (fast)
     - once with include_content=True (scrapes result pages for markdown)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from search_tool import FirecrawlTool


MEDIUM_URL = "https://medium.com/@mridulrao674385/language-modelling-on-mps-using-pytorch-044a2dfd9f62"
QUERY = "what is the current trending crypto"


def _short(s: Any, n: int = 600) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    s = s.strip()
    return s if len(s) <= n else (s[:n] + " ...[truncated]...")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("ERROR: FIRECRAWL_API_KEY is not set.", file=sys.stderr)
        return 2

    tool = FirecrawlTool(api_key=api_key)

    print("\n=== 1) lookup_tool(): Medium article ===")
    r1 = tool.lookup_tool(
        MEDIUM_URL,
        formats=["markdown"],  # you can add "html" too
        only_main_content=True,
        timeout_ms=120000,
        normalize=True,
    )
    if not r1.ok:
        print(f"lookup_tool FAILED: {r1.error} ({r1.exc_type})", file=sys.stderr)
        return 1

    # After the wrapper fix, markdown should reliably be here:
    data1 = r1.data if isinstance(r1.data, dict) else {}
    md1 = (data1.get("data") or {}).get("markdown")
    title1 = (data1.get("metadata") or {}).get("title")

    print(f"Title: {title1}")
    print(f"Markdown length: {len(md1 or '')}")
    print("Markdown preview:")
    print(_short(md1, 900))

    _write_json("out/firecrawl_lookup_medium.json", data1)
    print("Saved: out/firecrawl_lookup_medium.json")

    print("\n=== 2) search_tool(): query (include_content=False) ===")
    r2 = tool.search_tool(
        QUERY,
        limit=5,
        include_content=False,
        normalize=True,
    )
    if not r2.ok:
        print(f"search_tool FAILED: {r2.error} ({r2.exc_type})", file=sys.stderr)
        return 1

    data2 = r2.data if isinstance(r2.data, dict) else {}
    results2 = data2.get("results", [])
    print(f"Got {len(results2)} results")
    for i, it in enumerate(results2, 1):
        print(f"\n[{i}] {it.get('title')}\n    {it.get('url')}\n    {it.get('description')}")

    _write_json("out/firecrawl_search_crypto_fast.json", data2)
    print("Saved: out/firecrawl_search_crypto_fast.json")

    print("\n=== 3) search_tool(): query (include_content=True, formats=['markdown']) ===")
    r3 = tool.search_tool(
        QUERY,
        limit=3,  # keep this smaller because it scrapes each result
        include_content=True,
        content_formats=["markdown"],
        only_main_content=True,
        timeout_ms=60000,
        normalize=True,
    )
    if not r3.ok:
        print(f"search_tool(include_content=True) FAILED: {r3.error} ({r3.exc_type})", file=sys.stderr)
        return 1

    data3 = r3.data if isinstance(r3.data, dict) else {}
    results3 = data3.get("results", [])
    print(f"Got {len(results3)} results (with content)")
    for i, it in enumerate(results3, 1):
        content = it.get("content") or {}
        md = content.get("markdown")
        print(f"\n[{i}] {it.get('title')}\n    {it.get('url')}")
        print(f"    Markdown length: {len(md or '')}")
        print("    Markdown content preview:")
        print("    " + _short(md, 500).replace("\n", "\n    "))

    _write_json("out/firecrawl_search_crypto_with_content.json", data3)
    print("Saved: out/firecrawl_search_crypto_with_content.json")

    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
