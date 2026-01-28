from __future__ import annotations

import json
import os
import sys
import re
from typing import Optional
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from urllib.parse import urlparse

from integrations.firecrawl import FirecrawlTool, ToolResult


# -------------------------
# helpers
# -------------------------

def _clip(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    return s if len(s) <= n else s[:n]


def _is_http_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def clean_markdown_content(md: Optional[str]) -> Optional[str]:
    """
    Heuristic markdown cleaner focused on "page chrome" (Medium/nav/footer).
    Works on markdown output from Firecrawl.

    Goals:
    - Keep article body
    - Remove top nav / sign-in / sitemap / app prompts
    - Remove footer junk
    - Keep headings, paragraphs, code blocks
    """
    if md is None:
        return None

    s = md.strip()

    # 1) If we have an H1 ("# ..."), drop everything before the first H1.
    # Medium pages usually include tons of nav before the title.
    m = re.search(r"(?m)^\#\s+.+$", s)
    if m:
        s = s[m.start():].lstrip()

    # 2) Remove obvious boilerplate lines (case-insensitive, whole-line-ish)
    drop_line_patterns = [
        r"(?im)^\[sitemap\]\(.+\)\s*$",
        r"(?im)^\[open in app\]\(.+\)\s*$",
        r"(?im)^sign up\s*$",
        r"(?im)^\[sign in\]\(.+\)\s*$",
        r"(?im)^\[write\]\(.+\)\s*$",
        r"(?im)^\[search\]\(.+\)\s*$",
        r"(?im)^listen\s*$",
        r"(?im)^share\s*$",
        r"(?im)^\[medium logo\]\(.+\)\s*$",
    ]
    for pat in drop_line_patterns:
        s = re.sub(pat, "", s)

    # 3) Remove image-only lines like: ![](https://...)
    s = re.sub(r"(?m)^\!\[\]\([^)]+\)\s*$", "", s)

    # 4) Remove common Medium subscription/footer blocks
    #    (kept as broad multi-line removals)
    footer_block_patterns = [
        r"(?is)##\s+get\s+.*?inbox.*?(?:\n{2,}|\Z)",  # "Get X’s stories in your inbox"
        r"(?is)##\s+no\s+responses\s+yet.*?(?:\n{2,}|\Z)",
        r"(?is)^help\s*$.*\Z",  # often footer starts with Help/Status/About etc. (aggressive)
    ]
    for pat in footer_block_patterns:
        s = re.sub(pat, "", s).strip()

    # 5) Collapse excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s


def flatten_markdown_prose(md: Optional[str]) -> Optional[str]:
    if md is None:
        return None

    lines = md.splitlines()
    out = []
    in_code = False
    buffer = []

    def flush_buffer():
        if buffer:
            # join prose lines into one sentence
            out.append(" ".join(buffer).strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()

        # Toggle code block
        if stripped.startswith("```"):
            flush_buffer()
            in_code = not in_code
            out.append(line)  # keep fence
            continue

        if in_code:
            out.append(line)  # keep code as-is
            continue

        # Empty line → paragraph boundary
        if not stripped:
            flush_buffer()
            continue

        # Headings: keep but flatten
        if stripped.startswith("#"):
            flush_buffer()
            out.append(stripped)
            continue

        buffer.append(stripped)

    flush_buffer()

    return "\n".join(out)



# -------------------------
# response wrapper
# -------------------------

@dataclass
class LLMToolResponse:
    ok: bool
    data: Any = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
        }


# -------------------------
# LLM-facing tools
# -------------------------

class FirecrawlLLMTools:
    """
    LLM-facing facade over FirecrawlTool.
    Output is intentionally small + stable.
    """

    def __init__(self, fc: FirecrawlTool) -> None:
        self.fc = fc

    def web_search(self, query: str, *, k: int = 5) -> Dict[str, Any]:
        if not query.strip():
            return LLMToolResponse(
                False,
                None,
                {"message": "query is empty", "type": "BadInput"},
            ).to_dict()

        r: ToolResult = self.fc.search_tool(
            query,
            limit=k,
            include_content=False,
            normalize=True,
        )

        if not r.ok:
            return LLMToolResponse(
                False,
                None,
                {"message": r.error or "search failed", "type": r.exc_type},
            ).to_dict()

        payload = r.data if isinstance(r.data, dict) else {}
        results = payload.get("results", [])

        slim = []
        for it in results:
            if not isinstance(it, dict):
                continue
            url = it.get("url")
            if not url or not _is_http_url(url):
                continue
            slim.append(
                {
                    "title": it.get("title"),
                    "url": url,
                    "snippet": it.get("description"),
                }
            )

        return LLMToolResponse(True, {"results": slim}, None).to_dict()

    def web_fetch(
        self,
        url: str,
        *,
        format: Literal["markdown", "html", "text"] = "markdown",
        max_chars: int = 4000,
        only_main_content: bool = True,
        timeout_ms: int = 120000,
    ) -> Dict[str, Any]:
        if not _is_http_url(url):
            return LLMToolResponse(
                False,
                None,
                {"message": "invalid url", "type": "BadInput"},
            ).to_dict()

        r: ToolResult = self.fc.lookup_tool(
            url,
            formats=[format],
            only_main_content=only_main_content,
            timeout_ms=timeout_ms,
            normalize=True,
        )

        if not r.ok:
            return LLMToolResponse(
                False,
                None,
                {"message": r.error or "fetch failed", "type": r.exc_type},
            ).to_dict()

        data = r.data if isinstance(r.data, dict) else {}

        # title: try common places
        title = None
        if isinstance(data.get("metadata"), dict):
            title = data["metadata"].get("title")
        if not title and isinstance(data.get("data"), dict) and isinstance(data["data"].get("metadata"), dict):
            title = data["data"]["metadata"].get("title")

        def pick_content(d: Dict[str, Any], fmt: str) -> Optional[str]:
            # Most common normalized shape from your lookup_tool(normalize=True)
            if isinstance(d.get("data"), dict):
                v = d["data"].get(fmt)
                if isinstance(v, str) and v.strip():
                    return v

            # Some shapes keep content under raw
            if isinstance(d.get("raw"), dict):
                v = d["raw"].get(fmt)
                if isinstance(v, str) and v.strip():
                    return v

            # Sometimes nested: {"data": {"data": {"markdown": ...}}}
            if isinstance(d.get("data"), dict) and isinstance(d["data"].get("data"), dict):
                v = d["data"]["data"].get(fmt)
                if isinstance(v, str) and v.strip():
                    return v

            # Last resort: top-level
            v = d.get(fmt)
            if isinstance(v, str) and v.strip():
                return v

            return None

        content = pick_content(data, format)

        if format == "markdown" and isinstance(content, str):
            content = clean_markdown_content(content)

        return LLMToolResponse(
            True,
            {"url": url, "title": title, "content": _clip(content, int(max_chars))},
            None,
        ).to_dict()



# -------------------------
# HARD-CODED MAIN
# -------------------------

def main() -> int:
    """
    Hard-coded execution so you can inspect:
    - web_search output
    - web_fetch output
    and judge LLM suitability.
    """

    api_key = os.getenv("FIRECRAWL_API_KEY")
    fc = FirecrawlTool(api_key=api_key)
    tools = FirecrawlLLMTools(fc)

    # -------------------------
    # HARD-CODED INPUTS
    # -------------------------

    SEARCH_QUERY = "current trending cryptocurrencies"
    FETCH_URL = "https://www.reddit.com/r/AI_Agents/comments/1qojw8w/working_as_ai_engineer_is_wild/"

    print("\n=== LLM TOOL: web_search ===")
    search_out = tools.web_search(
        query=SEARCH_QUERY,
        k=int(os.getenv("SEARCH_TOOL_MAX_RESULTS", "5")),
    )
    print(json.dumps(search_out, indent=2, ensure_ascii=False))

    print("\n=== LLM TOOL: web_fetch ===")
    fetch_out = tools.web_fetch(
        url=FETCH_URL,
        format="markdown",
        max_chars=int(os.getenv("SEARCH_TOOL_MAX_CHARS", "4000")),
    )
    print(json.dumps(fetch_out, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
