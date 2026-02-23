from __future__ import annotations

import json
import os
import re
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse, urlunparse

from integrations.firecrawl import FirecrawlTool, ToolResult


# ---------------------------------------------------------------------------
# Env-var configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_CHARS: int = int(os.getenv("SEARCH_TOOL_MAX_CHARS", "10000"))
_DEFAULT_MAX_URLS: int  = int(os.getenv("SEARCH_TOOL_MAX_URLS",  "5"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sites that block all headless browsers — handled via native APIs / tricks
# ---------------------------------------------------------------------------

# Map: domain substring → handler name (used in web_fetch dispatch)
KNOWN_BLOCKED_DOMAINS: Dict[str, str] = {
    "reddit.com": "reddit",
    # Add more as you hit them, e.g.:
    # "twitter.com": "twitter",
    # "x.com": "twitter",
}


def _blocked_domain_handler(url: str) -> Optional[str]:
    """Return the handler name for a URL if its domain is known-blocked, else None."""
    try:
        host = urlparse(url).netloc.lower()
        for domain, handler in KNOWN_BLOCKED_DOMAINS.items():
            if domain in host:
                return handler
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Per-site native fetchers
# ---------------------------------------------------------------------------

def _fetch_reddit(url: str) -> Optional[str]:
    """
    Reddit blocks all headless scrapers but exposes an unofficial JSON API.
    Append .json to any reddit.com URL and it returns the full thread data.

    Works for:
      - Post threads:  /r/<sub>/comments/<id>/<slug>/
      - Subreddit:     /r/<sub>/
      - Search:        /r/<sub>/search?q=...

    Returns cleaned markdown of the post + top comments.
    """
    try:
        parsed = urlparse(url)
        # Ensure path ends with / before appending .json
        path = parsed.path.rstrip("/") + "/.json"
        json_url = urlunparse(parsed._replace(path=path, query=parsed.query or "limit=25&depth=2"))

        req = urllib.request.Request(
            json_url,
            headers={
                # Reddit requires a real User-Agent — "python-urllib" gets a 429
                "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())

        return _reddit_json_to_markdown(payload, url)

    except Exception as exc:
        return None  # caller will surface the error


def _reddit_json_to_markdown(payload: Any, source_url: str) -> Optional[str]:
    """Convert Reddit's .json response into readable markdown."""
    try:
        parts: List[str] = []

        # Reddit returns [post_listing, comments_listing] for thread URLs
        # or a single listing for subreddit/search pages
        if isinstance(payload, list) and len(payload) >= 1:
            post_listing = payload[0]
            comment_listing = payload[1] if len(payload) > 1 else None
        elif isinstance(payload, dict):
            post_listing = payload
            comment_listing = None
        else:
            return None

        # ── Post / OP ────────────────────────────────────────────────
        post_children = (
            post_listing.get("data", {}).get("children", [])
        )
        if post_children:
            post = post_children[0].get("data", {})
            title = post.get("title", "")
            author = post.get("author", "")
            score = post.get("score", 0)
            selftext = post.get("selftext", "").strip()
            subreddit = post.get("subreddit_name_prefixed", "")

            parts.append(f"# {title}")
            parts.append(f"**{subreddit}** · u/{author} · {score} upvotes\n")
            if selftext:
                parts.append(selftext)
            parts.append("")

        # ── Comments ─────────────────────────────────────────────────
        if comment_listing:
            comments = comment_listing.get("data", {}).get("children", [])
            parts.append("## Top Comments\n")
            for child in comments:
                c = child.get("data", {})
                if child.get("kind") != "t1":  # skip "more" placeholders
                    continue
                c_author = c.get("author", "")
                c_score = c.get("score", 0)
                c_body = (c.get("body") or "").strip()
                if not c_body or c_body == "[deleted]":
                    continue
                parts.append(f"**u/{c_author}** ({c_score} pts)")
                parts.append(c_body)
                parts.append("")

        return "\n".join(parts).strip() or None

    except Exception:
        return None


def clean_markdown_content(md: Optional[str]) -> Optional[str]:
    """
    Heuristic markdown cleaner focused on "page chrome" (nav/footer/Medium boilerplate).
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

    # 1) If we have an H1, drop everything before it (Medium nav etc.)
    m = re.search(r"(?m)^\#\s+.+$", s)
    if m:
        s = s[m.start():].lstrip()

    # 2) Drop obvious boilerplate lines
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

    # 3) Remove image-only lines: ![](https://...)
    s = re.sub(r"(?m)^\!\[\]\([^)]+\)\s*$", "", s)

    # 4) Remove common footer/subscription blocks
    footer_block_patterns = [
        r"(?is)##\s+get\s+.*?inbox.*?(?:\n{2,}|\Z)",
        r"(?is)##\s+no\s+responses\s+yet.*?(?:\n{2,}|\Z)",
        r"(?is)^help\s*$.*\Z",
    ]
    for pat in footer_block_patterns:
        s = re.sub(pat, "", s).strip()

    # 5) Collapse excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s or None


def strip_to_plaintext(text: Optional[str]) -> Optional[str]:
    """
    Strip markdown syntax and HTML tags down to plain prose before
    sending to an LLM filter. Goal is token reduction, not perfect cleaning.

    Removes: HTML tags, markdown links, images, bold/italic, code fences,
    inline code, horizontal rules, blockquote markers, heading hashes.
    Keeps: actual words, punctuation, newlines.
    """
    if not text:
        return None

    s = text

    # HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Markdown images: ![alt](url)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    # Markdown links: [text](url) → text
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    # Reference-style links: [text][ref] → text
    s = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", s)
    # Code fences (``` or ~~~)
    s = re.sub(r"```[\s\S]*?```", "[code block]", s)
    s = re.sub(r"~~~[\s\S]*?~~~", "[code block]", s)
    # Inline code
    s = re.sub(r"`[^`]+`", "[code]", s)
    # Heading hashes
    s = re.sub(r"(?m)^#{1,6}\s+", "", s)
    # Bold / italic (**, __, *, _)
    s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
    s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)
    # Horizontal rules
    s = re.sub(r"(?m)^[-*_]{3,}\s*$", "", s)
    # Blockquote markers
    s = re.sub(r"(?m)^>\s?", "", s)
    # HTML entities
    s = re.sub(r"&[a-zA-Z]+;", " ", s)
    s = re.sub(r"&#\d+;", " ", s)
    # Collapse whitespace
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip() or None


def flatten_markdown_prose(md: Optional[str]) -> Optional[str]:
    if md is None:
        return None

    lines = md.splitlines()
    out = []
    in_code = False
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            out.append(" ".join(buffer).strip())
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_buffer()
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if not stripped:
            flush_buffer()
            continue
        if stripped.startswith("#"):
            flush_buffer()
            out.append(stripped)
            continue
        buffer.append(stripped)

    flush_buffer()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Response wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMToolResponse:
    ok: bool
    data: Any = None
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


# ---------------------------------------------------------------------------
# LLM-facing tools
# ---------------------------------------------------------------------------

class FirecrawlLLMTools:
    """
    LLM-facing facade over FirecrawlTool.
    Output is intentionally small and stable for consumption by an LLM agent.
    """

    # Prompt used for content filtering — edit to change LLM behaviour
    _FILTER_SYSTEM_PROMPT = textwrap.dedent("""\
        You are a content extraction assistant for an AI agent.
        You will receive raw text scraped from a webpage and a context query.

        Your job:
        1. Extract ONLY information relevant to the query (or the full article body if no query).
        2. Remove all navigation, ads, cookie notices, footers, sign-up prompts, and boilerplate.
        3. Return a JSON object with this exact schema:
           {
             "title":      string | null,   // page or article title if identifiable
             "summary":    string,           // 2-3 sentence summary of the content
             "content":    string,           // clean prose, relevant sections only
             "key_points": [string]          // up to 5 bullet-point takeaways (empty list if none)
           }
        Return ONLY valid JSON. No markdown fences, no preamble.
    """)

    def __init__(self, fc: FirecrawlTool, *, openai_api_key: Optional[str] = None) -> None:
        self.fc = fc
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    def _llm_filter_content(
        self,
        raw_text: str,
        *,
        query: str = "",
        model: str = "gpt-4o-mini",
        max_input_chars: int = 12_000,
    ) -> Optional[Dict[str, Any]]:
        """
        Strip the raw text to plain prose, then call OpenAI to extract
        relevant, structured content.

        Returns a dict matching the schema in _FILTER_SYSTEM_PROMPT,
        or None if the call fails.
        """
        if not self._openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Pass openai_api_key=... to FirecrawlLLMTools "
                "or set the env var."
            )

        # Strip markdown/HTML before sending — reduces tokens significantly
        plain = strip_to_plaintext(raw_text) or raw_text
        plain = plain[:max_input_chars]  # hard cap

        user_msg = f"Query: {query or 'Extract the main article content.'}\n\n---\n\n{plain}"

        try:
            from openai import OpenAI  # lazy import

            client = OpenAI(api_key=self._openai_api_key)
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._FILTER_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw_json = resp.choices[0].message.content or ""
            return json.loads(raw_json)

        except Exception as exc:
            # Log but don't crash — caller decides what to do
            import logging
            logging.getLogger(__name__).warning("LLM filter failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # web_search
    # ------------------------------------------------------------------ #

    def web_search(
        self,
        query: str,
        *,
        k: int = _DEFAULT_MAX_URLS,
        include_content: bool = False,
        max_chars_per_result: int = _DEFAULT_MAX_CHARS,
        only_main_content: bool = True,
        timeout_ms: int = 60_000,
        llm_filter: bool = False,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Search the web for `query` and return the top-k results.

        Args:
            query:               Natural-language search query.
            k:                   Number of results to return.
            include_content:     If True, fetches full page markdown per result.
            max_chars_per_result: Clip raw content to this many chars per result.
            only_main_content:   Ask Firecrawl to strip nav/footer server-side.
            timeout_ms:          Per-request timeout.
            llm_filter:          If True, runs each result's content through
                                 OpenAI to produce structured filtered_content.
                                 Requires include_content=True and OPENAI_API_KEY.
            llm_model:           OpenAI model for filtering.

        Returns (ok=True):
            {
                "query": str,
                "total": int,
                "include_content": bool,
                "results": [
                    {
                        "index":            int,
                        "title":            str | None,
                        "url":              str,
                        "snippet":          str | None,
                        "content":          str | None,         # raw, when include_content=True
                        "filtered_content": { ... } | None,     # structured, when llm_filter=True
                    },
                    ...
                ]
            }
        """
        if not query.strip():
            return LLMToolResponse(
                False, None, {"message": "query is empty", "type": "BadInput"}
            ).to_dict()

        if llm_filter and not include_content:
            # Can't filter without content
            include_content = True

        r: ToolResult = self.fc.search_tool(
            query,
            limit=k,
            include_content=include_content,
            content_formats=["markdown"] if include_content else None,
            only_main_content=only_main_content,
            timeout_ms=timeout_ms,
            normalize=True,
        )

        if not r.ok:
            return LLMToolResponse(
                False, None, {"message": r.error or "search failed", "type": r.exc_type}
            ).to_dict()

        payload = r.data if isinstance(r.data, dict) else {}
        raw_results = payload.get("results", [])

        slim: List[Dict[str, Any]] = []
        for idx, it in enumerate(raw_results, start=1):
            if not isinstance(it, dict):
                continue
            url = it.get("url")
            if not url or not _is_http_url(url):
                continue

            raw_content: Optional[str] = None
            filtered_content: Optional[Dict[str, Any]] = None

            if include_content:
                c = it.get("content") or {}
                raw_md = c.get("markdown_filtered") or c.get("markdown")
                if raw_md:
                    raw_content = _clip(clean_markdown_content(raw_md), max_chars_per_result)

                if llm_filter and raw_md:
                    filtered_content = self._llm_filter_content(
                        raw_md, query=query, model=llm_model
                    )

            slim.append(
                {
                    "index":            idx,
                    "title":            it.get("title"),
                    "url":              url,
                    "snippet":          it.get("description"),
                    "content":          raw_content,
                    "filtered_content": filtered_content,
                }
            )

        return LLMToolResponse(
            True,
            {
                "query":           query,
                "total":           len(slim),
                "include_content": include_content,
                "results":         slim,
            },
            None,
        ).to_dict()

    # ------------------------------------------------------------------ #
    # web_fetch
    # ------------------------------------------------------------------ #

    def web_fetch(
        self,
        url: str,
        *,
        query: str = "",
        format: Literal["markdown", "html", "text"] = "markdown",
        max_chars: int = _DEFAULT_MAX_CHARS,
        only_main_content: bool = True,
        timeout_ms: int = 120_000,
        llm_filter: bool = False,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Scrape a single URL and return its cleaned content.

        For sites that block headless scrapers (e.g. Reddit), automatically
        falls back to native API fetchers before trying Firecrawl.

        Args:
            url:              The URL to fetch.
            query:            What you're looking for — used as context for
                              LLM filtering (ignored if llm_filter=False).
            format:           Firecrawl format to request.
            max_chars:        Clip raw content to this many chars.
            only_main_content: Strip nav/footer server-side.
            timeout_ms:       Scrape timeout.
            llm_filter:       If True, run OpenAI over the raw content and
                              return structured filtered_content alongside
                              the raw clipped content.
            llm_model:        OpenAI model for filtering.

        Returns (ok=True):
            {
                "url":              str,
                "title":            str | None,
                "content":          str | None,     # raw, clipped to max_chars
                "filtered_content": {               # only when llm_filter=True
                    "title":      str | None,
                    "summary":    str,
                    "content":    str,
                    "key_points": [str],
                } | None,
                "source": "firecrawl" | "native_api",
            }
        """
        if not _is_http_url(url):
            return LLMToolResponse(
                False, None, {"message": "invalid url", "type": "BadInput"}
            ).to_dict()

        # ── Dispatch known-blocked domains to native fetchers ─────────
        handler = _blocked_domain_handler(url)
        if handler == "reddit":
            raw_content = _fetch_reddit(url)
            if raw_content is None:
                return LLMToolResponse(
                    False, None,
                    {"message": "Reddit fetch failed. Thread may be private or deleted.", "type": "NativeFetchError"}
                ).to_dict()

            filtered_content = None
            if llm_filter:
                filtered_content = self._llm_filter_content(
                    raw_content, query=query, model=llm_model
                )

            return LLMToolResponse(
                True,
                {
                    "url":              url,
                    "title":            (filtered_content or {}).get("title"),
                    "content":          _clip(raw_content, int(max_chars)),
                    "filtered_content": filtered_content,
                    "source":           "native_api",
                },
                None,
            ).to_dict()

        # ── Standard Firecrawl path ───────────────────────────────────
        r: ToolResult = self.fc.lookup_tool(
            url,
            formats=[format],
            only_main_content=only_main_content,
            timeout_ms=timeout_ms,
            normalize=True,
        )

        if not r.ok:
            return LLMToolResponse(
                False, None, {"message": r.error or "fetch failed", "type": r.exc_type}
            ).to_dict()

        data = r.data if isinstance(r.data, dict) else {}

        title: Optional[str] = None
        for loc in (data.get("metadata"), (data.get("data") or {}).get("metadata")):
            if isinstance(loc, dict):
                title = loc.get("title")
                if title:
                    break

        def _pick_content(d: Dict[str, Any], fmt: str) -> Optional[str]:
            candidates = [
                (d.get("data") or {}).get(fmt),
                (d.get("raw") or {}).get(fmt),
                ((d.get("data") or {}).get("data") or {}).get(fmt),
                d.get(fmt),
            ]
            for v in candidates:
                if isinstance(v, str) and v.strip():
                    return v
            return None

        raw_content = _pick_content(data, format)

        if format == "markdown" and isinstance(raw_content, str):
            raw_content = clean_markdown_content(raw_content)

        filtered_content = None
        if llm_filter and raw_content:
            filtered_content = self._llm_filter_content(
                raw_content, query=query, model=llm_model
            )
            # Prefer LLM-extracted title over metadata title if available
            if not title and filtered_content:
                title = filtered_content.get("title")

        return LLMToolResponse(
            True,
            {
                "url":              url,
                "title":            title,
                "content":          _clip(raw_content, int(max_chars)),
                "filtered_content": filtered_content,
                "source":           "firecrawl",
            },
            None,
        ).to_dict()


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_result(out: Dict[str, Any], *, show_content_preview: bool = True) -> None:
    """Pretty-print a tool result for inspection."""
    if not out["ok"]:
        print(f"  ❌ ERROR: {out['error']}")
        return

    data = out["data"] or {}

    # web_search
    if "results" in data:
        results = data["results"]
        print(f"  ✅ {data.get('total', len(results))} results  "
              f"(include_content={data.get('include_content')})")
        for r in results:
            print(f"\n  [{r['index']}] {r.get('title') or '(no title)'}")
            print(f"       {r['url']}")
            if r.get("snippet"):
                print(f"       snippet: {textwrap.shorten(r['snippet'], 120)}")
            if show_content_preview and r.get("content"):
                preview = textwrap.shorten(r["content"], 300)
                print(f"       content: {preview}")
        return

    # web_fetch
    if "content" in data:
        print(f"  ✅ {data.get('url')}")
        print(f"     title:   {data.get('title')}")
        content = data.get("content") or ""
        if show_content_preview and content:
            print(f"     content: {textwrap.shorten(content, 400)}")
            print(f"     length:  {len(content)} chars")
        return

    # fallback: raw dump
    print(json.dumps(data, indent=2, ensure_ascii=False))


def run_tests(tools: FirecrawlLLMTools) -> None:
    """
    Exercise all meaningful modes of web_search and web_fetch.
    Edit CASES below to add your own scenarios.
    """

    SEARCH_CASES = [
        {
            "label": "web_search — structured filtered content (k=2)",
            "query": "LiveKit Python SDK audio track example",
            "k": 2,
            "include_content": True,
            "llm_filter": True,
        },
    ]

    FETCH_CASES = [
        {
            "label": "web_fetch — Reddit thread, raw only",
            "url": "https://www.reddit.com/r/AI_Agents/comments/1qojw8w/working_as_ai_engineer_is_wild/",
            "query": "",
            "llm_filter": True,
        },
    ]

    def _print_result(out: Dict[str, Any]) -> None:
        if not out["ok"]:
            print(f"  ❌ ERROR: {out['error']}")
            return

        data = out["data"] or {}

        # web_search
        if "results" in data:
            results = data["results"]
            print(f"  ✅ {data.get('total', len(results))} results  "
                  f"(include_content={data.get('include_content')})")
            for r in results:
                print(f"\n  [{r['index']}] {r.get('title') or '(no title)'}")
                print(f"       {r['url']}")
                if r.get("snippet"):
                    print(f"       snippet:  {textwrap.shorten(r['snippet'], 120)}")
                if r.get("content"):
                    print(f"       content:  {textwrap.shorten(r['content'], 250)}")
                if r.get("filtered_content"):
                    fc = r["filtered_content"]
                    print(f"       summary:  {textwrap.shorten(fc.get('summary',''), 200)}")
                    for pt in (fc.get("key_points") or [])[:3]:
                        print(f"         • {pt}")
            return

        # web_fetch
        if "content" in data:
            print(f"  ✅ [{data.get('source')}] {data.get('url')}")
            print(f"     title:   {data.get('title')}")
            if data.get("content"):
                print(f"     content: {textwrap.shorten(data['content'], 300)} "
                      f"({len(data['content'])} chars)")
            fc = data.get("filtered_content")
            if fc:
                print(f"\n     — filtered_content —")
                print(f"     title:      {fc.get('title')}")
                print(f"     summary:    {textwrap.shorten(fc.get('summary',''), 250)}")
                print(f"     content:    {textwrap.shorten(fc.get('content',''), 300)}")
                for pt in (fc.get("key_points") or []):
                    print(f"       • {pt}")
            return

        print(json.dumps(data, indent=2, ensure_ascii=False))

    for case in SEARCH_CASES:
        _print_header(case["label"])
        t0 = time.perf_counter()
        out = tools.web_search(
            query=case["query"],
            k=case.get("k", 5),
            include_content=case.get("include_content", False),
            max_chars_per_result=case.get("max_chars_per_result", 2000),
            llm_filter=case.get("llm_filter", False),
        )
        print(f"  ⏱  {time.perf_counter() - t0:.2f}s")
        _print_result(out)
        if os.getenv("DEBUG_RAW"):
            print(json.dumps(out, indent=2, ensure_ascii=False))

    for case in FETCH_CASES:
        _print_header(case["label"])
        t0 = time.perf_counter()
        out = tools.web_fetch(
            url=case["url"],
            query=case.get("query", ""),
            max_chars=case.get("max_chars", 4000),
            llm_filter=case.get("llm_filter", False),
        )
        print(f"  ⏱  {time.perf_counter() - t0:.2f}s")
        _print_result(out)
        if os.getenv("DEBUG_RAW"):
            print(json.dumps(out, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("ERROR: FIRECRAWL_API_KEY not set")
        return 1

    fc = FirecrawlTool(api_key=api_key)
    tools = FirecrawlLLMTools(fc, openai_api_key=os.getenv("OPENAI_API_KEY"))
    run_tests(tools)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())