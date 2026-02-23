"""
Firecrawl tool wrapper for LLM agents.

This is a utility class over the Firecrawl SDK.
A separate layer should be built on top for LLM tool-calling adapters
(LangChain @tool, OpenAI function spec, etc.).

LLM Filtering
─────────────
Both `search_tool` and `lookup_tool` accept an optional `llm_filter` callable:

    Signature: (text: str, query: str) -> str

Pass any callable that takes raw markdown + a query hint and returns
cleaned/relevant text. A ready-made `openai_filter()` factory is provided,
but you can wire in Anthropic, Azure OpenAI, local Ollama, etc.

    from firecrawl_tools import FirecrawlTool, openai_filter

    tool = FirecrawlTool(
        llm_filter=openai_filter(model="gpt-4o-mini")  # optional, can be None
    )
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union

from dotenv import load_dotenv
from firecrawl import Firecrawl, FirecrawlApp

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Type alias for the filter callable
LLMFilter = Callable[[str, str], str]  # (text, query) -> filtered_text

# Max chars sent to the LLM filter to stay within token budget
_FILTER_CHAR_LIMIT = 12_000


# ---------------------------------------------------------------------------
# Built-in LLM filter factory (OpenAI) — optional, swap freely
# ---------------------------------------------------------------------------

def openai_filter(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_tokens: int = 2000,
) -> LLMFilter:
    """
    Returns an LLM filter backed by OpenAI chat completions.

    Usage:
        tool = FirecrawlTool(llm_filter=openai_filter(model="gpt-4o-mini"))

    Swap for any other LLM by writing your own (text, query) -> str callable.
    """
    _api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _filter(text: str, query: str) -> str:
        try:
            from openai import OpenAI  # lazy import — only required if filter is used

            client = OpenAI(api_key=_api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a content extraction assistant. "
                            "Given raw markdown scraped from a webpage and a user query, "
                            "extract ONLY the information relevant to the query. "
                            "Remove navigation, ads, cookie banners, footers, and unrelated sections. "
                            "Return clean, concise markdown. "
                            "If the page has no relevant content, return an empty string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\n---\n\n{text[:_FILTER_CHAR_LIMIT]}",
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM filter failed, returning raw text. Error: %s", exc)
            return text

    return _filter


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    exc_type: Optional[str] = None

    @staticmethod
    def success(data: Any) -> "ToolResult":
        return ToolResult(ok=True, data=data)

    @staticmethod
    def fail(msg: str, exc: Optional[BaseException] = None) -> "ToolResult":
        return ToolResult(
            ok=False,
            data=None,
            error=msg,
            exc_type=type(exc).__name__ if exc else None,
        )


# ---------------------------------------------------------------------------
# FirecrawlTool
# ---------------------------------------------------------------------------

class FirecrawlTool:
    """
    Thin, testable wrapper over the Firecrawl SDK.

    Canonical agent-facing tools:
      - search_tool(query, ...) : web search; optionally scrape + filter results
      - lookup_tool(url, ...)   : scrape a specific URL; optionally filter content

    All other methods (crawl, map, batch_scrape, agent_tool, ...) are lower-level
    helpers exposed for completeness.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[Firecrawl] = None,
        app: Optional[Any] = None,
        llm_filter: Optional[LLMFilter] = None,
    ) -> None:
        """
        Args:
            api_key:    Firecrawl API key. Falls back to FIRECRAWL_API_KEY env var.
            client:     Pre-built Firecrawl instance (for testing / DI).
            app:        Pre-built FirecrawlApp instance (for testing / DI).
            llm_filter: Optional callable (text, query) -> str.
                        When provided it is used by default in search_tool and
                        lookup_tool unless overridden per-call.
                        Pass None (default) to skip filtering entirely.
                        Use the built-in `openai_filter()` factory or supply your own.
        """
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key and client is None:
            raise ValueError(
                "Missing Firecrawl API key. Provide api_key=... or set FIRECRAWL_API_KEY."
            )

        self.client: Firecrawl = client or Firecrawl(api_key=api_key)
        self.app: Optional[Any] = app or (FirecrawlApp(api_key=api_key) if api_key else None)
        self._default_llm_filter: Optional[LLMFilter] = llm_filter

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _to_dict(self, x: Any) -> Any:
        """Recursively convert Firecrawl SDK objects to plain Python dicts."""
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: self._to_dict(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._to_dict(v) for v in x]
        if hasattr(x, "model_dump"):  # pydantic v2
            return self._to_dict(x.model_dump())
        if hasattr(x, "dict"):  # pydantic v1
            return self._to_dict(x.dict())
        if hasattr(x, "__dict__"):
            return self._to_dict(dict(x.__dict__))
        return x

    def _resolve_filter(self, per_call: Optional[Union[LLMFilter, bool]]) -> Optional[LLMFilter]:
        """
        Resolve which LLM filter to use for a single call.

        per_call semantics:
          callable → use it directly for this call only
          True     → force the instance default (raises if none configured)
          False    → explicitly disable filtering for this call
          None     → use instance default if one is set, otherwise no filtering
        """
        if callable(per_call):
            return per_call
        if per_call is True:
            if self._default_llm_filter is None:
                raise ValueError(
                    "llm_filter=True but no default filter configured on FirecrawlTool. "
                    "Pass llm_filter=openai_filter() to the constructor, or supply a callable per-call."
                )
            return self._default_llm_filter
        if per_call is False:
            return None  # explicit opt-out, ignore instance default
        # per_call is None → use instance default (which may also be None)
        return self._default_llm_filter

    def _apply_filter(
        self,
        text: Optional[str],
        query: str,
        llm_filter: Optional[LLMFilter],
    ) -> Optional[str]:
        """Run filter if provided and text is non-empty."""
        if not text or llm_filter is None:
            return text
        logger.debug("Applying LLM filter for query=%r (%d chars)", query, len(text))
        return llm_filter(text, query)

    def _extract_scrape_payload(self, raw: Any) -> Dict[str, Any]:
        """
        Normalize Firecrawl scrape responses to a stable shape:

          {
            "data": {...formats...} | None,
            "metadata": {...} | None,
            "raw": <original dict-safe payload>
          }

        Handles both response shapes the SDK may return:
          A) {"success": bool, "data": {...}, "metadata": {...}}
          B) {"markdown": "...", "metadata": {...}, ...}  (formats at top-level)
        """
        raw_obj = raw if isinstance(raw, dict) else {"value": raw}

        data_obj = raw_obj.get("data")
        metadata_obj = raw_obj.get("metadata")

        if isinstance(data_obj, dict):
            return {
                "data": data_obj,
                "metadata": metadata_obj if isinstance(metadata_obj, dict) else None,
                "raw": raw_obj,
            }

        # Fallback: formats live at top-level
        known_format_keys = (
            "markdown", "html", "rawHtml", "raw_html",
            "links", "images", "screenshot", "json",
            "summary", "actions", "warning", "change_tracking", "branding",
        )
        extracted: Dict[str, Any] = {
            k: raw_obj[k] for k in known_format_keys if raw_obj.get(k) is not None
        }

        return {
            "data": extracted or None,
            "metadata": metadata_obj if isinstance(metadata_obj, dict) else None,
            "raw": raw_obj,
        }

    # ------------------------------------------------------------------ #
    # CANONICAL AGENT TOOLS
    # ------------------------------------------------------------------ #

    def search_tool(
        self,
        query: str,
        *,
        limit: int = 5,
        include_content: bool = True,
        content_formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        timeout_ms: int = 60_000,
        llm_filter: Optional[Union[LLMFilter, bool]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        AGENT TOOL #1 — Web search returning content from N URLs.

        Args:
            query:           Natural-language search query.
            limit:           Number of URLs to fetch (user-defined N).
            include_content: If True, Firecrawl scrapes each result and returns
                             its content alongside url/title/description.
            content_formats: Formats to request per result (default: ["markdown"]).
            only_main_content: Strip nav/ads at Firecrawl level before filtering.
            timeout_ms:      Per-request timeout passed to Firecrawl.
            llm_filter:      Controls LLM post-processing of scraped content:
                               - None      → use instance default (if any)
                               - True      → force instance default (error if none configured)
                               - False     → skip filtering even if instance default is set
                               - callable  → use this filter for this call only
            normalize:       If True, returns structured results list.
                             If False, returns raw Firecrawl payload.

        Returns:
            ToolResult.data = {
                "query":           str,
                "limit":           int,
                "include_content": bool,
                "results": [
                    {
                        "title":       str | None,
                        "url":         str | None,
                        "description": str | None,
                        "content": {
                            "markdown":          str | None,  # raw from Firecrawl
                            "markdown_filtered": str | None,  # after LLM filter (if used)
                        } | None,
                        "source": "firecrawl",
                        "extra": {...},
                    },
                    ...
                ],
                "raw": <original Firecrawl payload>,
            }
        """
        if not query or not query.strip():
            return ToolResult.fail("search_tool: query is empty")

        resolved_filter = self._resolve_filter(llm_filter)

        try:
            scrape_options = None
            if include_content:
                scrape_options = {
                    "formats": content_formats or ["markdown"],
                    "only_main_content": bool(only_main_content),
                    "timeout": int(timeout_ms),
                }

            raw = self.client.search(
                query=query,
                limit=int(limit),
                scrape_options=scrape_options,
                **kwargs,
            )
            raw_obj = self._to_dict(raw)

            if not normalize:
                return ToolResult.success(raw_obj)

            results = self._normalize_search_results(raw_obj, query=query, llm_filter=resolved_filter)
            return ToolResult.success(
                {
                    "query": query,
                    "limit": int(limit),
                    "include_content": bool(include_content),
                    "results": results,
                    "raw": raw_obj,
                }
            )
        except Exception as e:
            return ToolResult.fail(f"search_tool failed for query={query!r}", e)

    def lookup_tool(
        self,
        url: str,
        *,
        query: str = "",
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        timeout_ms: int = 120_000,
        llm_filter: Optional[Union[LLMFilter, bool]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        AGENT TOOL #2 — Scrape a single URL and return filtered content.

        Args:
            url:           The URL to scrape.
            query:         Context hint for LLM filtering — what are you looking for?
                           Falls back to general noise-removal if empty.
            formats:       Firecrawl formats to request (default: ["markdown"]).
            only_main_content: Strip nav/ads at Firecrawl level.
            timeout_ms:    Scrape timeout.
            llm_filter:    Same semantics as search_tool.llm_filter.
            normalize:     If True, returns structured payload.

        Returns:
            ToolResult.data = {
                "url":  str,
                "data": {
                    "markdown":          str | None,  # raw from Firecrawl
                    "markdown_filtered": str | None,  # after LLM filter (if used)
                    ...other requested formats...
                } | None,
                "metadata": {...} | None,
                "raw": <original Firecrawl payload>,
            }
        """
        if not url or not url.strip():
            return ToolResult.fail("lookup_tool: url is empty")

        resolved_filter = self._resolve_filter(llm_filter)

        res = self.scrape(
            url,
            formats=formats or ["markdown"],
            only_main_content=only_main_content,
            timeout_ms=timeout_ms,
            **kwargs,
        )
        if not res.ok:
            return res
        if not normalize:
            return res

        payload = self._extract_scrape_payload(res.data)
        data = dict(payload["data"]) if isinstance(payload["data"], dict) else {}

        # Apply LLM filter to markdown if present
        raw_md = data.get("markdown") or ""
        if resolved_filter is not None:
            filter_query = query.strip() or "Extract the main content, removing navigation, ads, and boilerplate."
            data["markdown_filtered"] = self._apply_filter(raw_md, filter_query, resolved_filter)
        else:
            data["markdown_filtered"] = None

        return ToolResult.success(
            {
                "url": url,
                "data": data or None,
                "metadata": payload["metadata"],
                "raw": payload["raw"],
            }
        )

    # ------------------------------------------------------------------ #
    # Normalizers
    # ------------------------------------------------------------------ #

    def _normalize_search_results(
        self,
        raw: Any,
        *,
        query: str = "",
        llm_filter: Optional[LLMFilter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Normalize Firecrawl search results to a stable list shape.
        Applies LLM filter to markdown content when a filter is provided.
        """
        items: List[Any] = []

        if isinstance(raw, dict):
            if isinstance(raw.get("web"), list):
                items = raw["web"]
            elif isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("web"), list):
                items = raw["data"]["web"]
            else:
                for k in ("results", "items", "data"):
                    v = raw.get(k)
                    if isinstance(v, list):
                        items = v
                        break
        elif isinstance(raw, list):
            items = raw

        normalized: List[Dict[str, Any]] = []
        content_keys = ("markdown", "html", "rawHtml", "raw_html", "links", "images", "screenshot", "json", "summary")

        for it in items:
            it = it if isinstance(it, dict) else {"value": it}
            md = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}

            url = (
                md.get("url") or it.get("url") or it.get("link")
                or it.get("href") or md.get("source_url")
            )
            title = md.get("title") or it.get("title") or it.get("name")
            desc = (
                md.get("description") or it.get("description")
                or it.get("snippet") or it.get("summary")
            )

            content: Dict[str, Any] = {k: it[k] for k in content_keys if it.get(k) is not None}

            # LLM filter on markdown
            raw_md = content.get("markdown") or ""
            if llm_filter is not None:
                filter_query = query or title or "Extract the relevant content."
                content["markdown_filtered"] = self._apply_filter(raw_md, filter_query, llm_filter)
            else:
                content["markdown_filtered"] = None

            extra = {k: v for k, v in it.items() if k not in ("metadata", *content_keys)}

            normalized.append(
                {
                    "title": title,
                    "url": url,
                    "description": desc,
                    "content": content or None,
                    "source": "firecrawl",
                    "extra": extra,
                }
            )

        return normalized

    # ------------------------------------------------------------------ #
    # Lower-level SDK wrappers
    # ------------------------------------------------------------------ #

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        only_main_content: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Scrape a single URL (raw SDK call, no filtering)."""
        if not url or not url.strip():
            return ToolResult.fail("scrape: url is empty")
        try:
            params: Dict[str, Any] = {}
            if formats is not None:
                params["formats"] = formats
            if only_main_content is not None:
                params["only_main_content"] = only_main_content
            if timeout_ms is not None:
                params["timeout"] = int(timeout_ms)
            params.update(kwargs)

            data = self.client.scrape(url, **params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"scrape failed for url={url!r}", e)

    def crawl(
        self,
        url: str,
        *,
        limit: Optional[int] = None,
        scrape_options: Optional[Dict[str, Any]] = None,
        sitemap: Optional[str] = None,
        poll_interval: Optional[int] = None,
        timeout_s: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            params: Dict[str, Any] = {"url": url}
            if limit is not None:
                params["limit"] = int(limit)
            if scrape_options is not None:
                params["scrape_options"] = scrape_options
            if sitemap is not None:
                params["sitemap"] = sitemap
            if poll_interval is not None:
                params["poll_interval"] = int(poll_interval)
            if timeout_s is not None:
                params["timeout"] = int(timeout_s)
            params.update(kwargs)

            return ToolResult.success(self._to_dict(self.client.crawl(**params)))
        except Exception as e:
            return ToolResult.fail(f"crawl failed for url={url!r}", e)

    def start_crawl(self, url: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        try:
            params: Dict[str, Any] = {"url": url}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.client.start_crawl(**params)))
        except Exception as e:
            return ToolResult.fail(f"start_crawl failed for url={url!r}", e)

    def get_crawl_status(self, crawl_id: str, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult.success(self._to_dict(self.client.get_crawl_status(crawl_id, **kwargs)))
        except Exception as e:
            return ToolResult.fail(f"get_crawl_status failed for id={crawl_id!r}", e)

    def cancel_crawl(self, crawl_id: str) -> ToolResult:
        try:
            return ToolResult.success(self._to_dict(self.client.cancel_crawl(crawl_id)))
        except Exception as e:
            return ToolResult.fail(f"cancel_crawl failed for id={crawl_id!r}", e)

    def get_crawl_status_page(self, next_url: str) -> ToolResult:
        try:
            return ToolResult.success(self._to_dict(self.client.get_crawl_status_page(next_url)))
        except Exception as e:
            return ToolResult.fail("get_crawl_status_page failed", e)

    def map(self, url: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        try:
            params: Dict[str, Any] = {"url": url}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.client.map(**params)))
        except Exception as e:
            return ToolResult.fail(f"map failed for url={url!r}", e)

    def search(self, query: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        """Raw Firecrawl search (no normalization or filtering). Kept for backward-compat."""
        try:
            params: Dict[str, Any] = {"query": query}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.client.search(**params)))
        except Exception as e:
            return ToolResult.fail(f"search failed for query={query!r}", e)

    def batch_scrape(
        self,
        urls: Sequence[str],
        *,
        formats: Optional[List[str]] = None,
        poll_interval: Optional[int] = None,
        wait_timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            params: Dict[str, Any] = {}
            if formats is not None:
                params["formats"] = formats
            if poll_interval is not None:
                params["poll_interval"] = int(poll_interval)
            if wait_timeout is not None:
                params["wait_timeout"] = int(wait_timeout)
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.client.batch_scrape(list(urls), **params)))
        except Exception as e:
            return ToolResult.fail("batch_scrape failed", e)

    def start_batch_scrape(
        self,
        urls: Sequence[str],
        *,
        formats: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            params: Dict[str, Any] = {}
            if formats is not None:
                params["formats"] = formats
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.client.start_batch_scrape(list(urls), **params)))
        except Exception as e:
            return ToolResult.fail("start_batch_scrape failed", e)

    def get_batch_scrape_status(self, batch_id: str, **kwargs: Any) -> ToolResult:
        try:
            return ToolResult.success(self._to_dict(self.client.get_batch_scrape_status(batch_id, **kwargs)))
        except Exception as e:
            return ToolResult.fail(f"get_batch_scrape_status failed for id={batch_id!r}", e)

    def get_batch_scrape_status_page(self, next_url: str) -> ToolResult:
        try:
            return ToolResult.success(self._to_dict(self.client.get_batch_scrape_status_page(next_url)))
        except Exception as e:
            return ToolResult.fail("get_batch_scrape_status_page failed", e)

    # ------------------------------------------------------------------ #
    # Agent (FirecrawlApp-based)
    # ------------------------------------------------------------------ #

    def agent_tool(
        self,
        prompt: str,
        *,
        urls: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        schema: Optional[Union[Type[T], Dict[str, Any]]] = None,
        timeout_s: int = 180,
        poll_interval_s: float = 2.0,
        **kwargs: Any,
    ) -> ToolResult:
        """Agent mode (search + navigate + extract) via FirecrawlApp."""
        if not prompt or not prompt.strip():
            return ToolResult.fail("agent_tool: prompt is empty")
        if self.app is None:
            return ToolResult.fail(
                "agent_tool: FirecrawlApp is not available. "
                "Install/upgrade the Firecrawl SDK or pass app=... to the constructor."
            )
        try:
            start_params: Dict[str, Any] = {"prompt": prompt}
            if urls is not None:
                start_params["urls"] = list(urls)
            if model is not None:
                start_params["model"] = model
            if schema is not None:
                start_params["schema"] = schema
            start_params.update(kwargs)

            job = self.app.start_agent(**start_params)
            job_obj = self._to_dict(job)
            job_id = (
                job_obj.get("id") or job_obj.get("jobId") or job_obj.get("job_id")
                or getattr(job, "id", None)
            )
            if not job_id:
                return ToolResult.fail(
                    f"agent_tool: could not determine job id from start_agent response: {job_obj}"
                )

            deadline = time.time() + float(timeout_s)
            last_status: Optional[Dict[str, Any]] = None

            while time.time() < deadline:
                status_obj = self._to_dict(self.app.get_agent_status(job_id))
                last_status = status_obj
                st = status_obj.get("status") or status_obj.get("state")

                if st == "completed":
                    return ToolResult.success(
                        {
                            "prompt": prompt,
                            "job_id": job_id,
                            "status": "completed",
                            "data": status_obj.get("data"),
                            "creditsUsed": status_obj.get("creditsUsed") or status_obj.get("credits_used"),
                            "expiresAt": status_obj.get("expiresAt") or status_obj.get("expires_at"),
                            "raw": status_obj,
                        }
                    )
                if st == "failed":
                    return ToolResult.fail(
                        f"Firecrawl agent failed: {status_obj.get('error') or 'unknown error'}"
                    )

                time.sleep(float(poll_interval_s))

            return ToolResult.fail(
                f"agent_tool: timed out after {timeout_s}s. last_status={last_status}"
            )
        except Exception as e:
            return ToolResult.fail("agent_tool failed", e)

    def agent(
        self,
        *,
        prompt: str,
        urls: Optional[Sequence[str]] = None,
        schema: Optional[Union[Type[T], Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Blocking agent() call if your SDK version supports it."""
        if self.app is None:
            return ToolResult.fail(
                "agent: FirecrawlApp is not available. Install/upgrade SDK or pass app=... to constructor."
            )
        try:
            params: Dict[str, Any] = {"prompt": prompt}
            if urls is not None:
                params["urls"] = list(urls)
            if schema is not None:
                params["schema"] = schema
            if model is not None:
                params["model"] = model
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.app.agent(**params)))
        except Exception as e:
            return ToolResult.fail("agent failed", e)

    def start_agent(self, *, prompt: str, urls: Optional[Sequence[str]] = None, **kwargs: Any) -> ToolResult:
        if self.app is None:
            return ToolResult.fail("start_agent: FirecrawlApp is not available.")
        try:
            params: Dict[str, Any] = {"prompt": prompt}
            if urls is not None:
                params["urls"] = list(urls)
            params.update(kwargs)
            return ToolResult.success(self._to_dict(self.app.start_agent(**params)))
        except Exception as e:
            return ToolResult.fail("start_agent failed", e)

    def get_agent_status(self, agent_id: str, **kwargs: Any) -> ToolResult:
        if self.app is None:
            return ToolResult.fail("get_agent_status: FirecrawlApp is not available.")
        try:
            return ToolResult.success(self._to_dict(self.app.get_agent_status(agent_id, **kwargs)))
        except Exception as e:
            return ToolResult.fail(f"get_agent_status failed for id={agent_id!r}", e)


# ---------------------------------------------------------------------------
# Quick smoke test  (python firecrawl_tools.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Wire up with an LLM filter at construction time (optional)
    tool = FirecrawlTool(llm_filter=openai_filter(model="gpt-4o-mini"))

    # ── Tool 1: search ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("search_tool — pgvector vs pinecone, 3 results, with filtering")
    print("=" * 60)

    res = tool.search_tool(
        "pgvector vs pinecone for production RAG",
        limit=3,
        include_content=True,
        # llm_filter=False  ← uncomment to skip filtering for this call only
    )
    if res.ok:
        for i, r in enumerate(res.data["results"], 1):
            content = r.get("content") or {}
            # prefer filtered content, fall back to raw markdown
            text = content.get("markdown_filtered") or content.get("markdown") or ""
            print(f"\n[{i}] {r['title']}  —  {r['url']}")
            print(text[:400])
    else:
        print("ERROR:", res.error)

    # ── Tool 2: lookup ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("lookup_tool — Firecrawl docs auth, with filtering")
    print("=" * 60)

    res2 = tool.lookup_tool(
        "https://docs.firecrawl.dev/api-reference/introduction",
        query="what authentication method does the API use?",
        # llm_filter=False  ← uncomment to skip filtering for this call only
    )
    if res2.ok:
        data = res2.data.get("data") or {}
        text = data.get("markdown_filtered") or data.get("markdown") or ""
        print(text[:800])
    else:
        print("ERROR:", res2.error)