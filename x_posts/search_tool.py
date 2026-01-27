from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from dotenv import load_dotenv

# Firecrawl v2 SDK (scrape/search/crawl/map/etc.)
from firecrawl import Firecrawl

# FirecrawlApp is still the class shown in Agent docs/examples.
# (Depending on your installed SDK version, FirecrawlApp may be available.)
try:
    from firecrawl import FirecrawlApp  # type: ignore
except Exception:  # pragma: no cover
    FirecrawlApp = None  # type: ignore

load_dotenv()

T = TypeVar("T")


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


class FirecrawlTool:
    """
    Thin, testable wrapper over Firecrawl SDK.

    Canonical agent-facing tools:
      - search_tool(query, ...) : web search; optionally scrape results for content
      - lookup_tool(url, ...)   : scrape a specific URL for main content
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client: Optional[Firecrawl] = None,
        app: Optional[Any] = None,  # FirecrawlApp
    ) -> None:
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key and client is None:
            raise ValueError(
                "Missing Firecrawl API key. Provide api_key=... or set FIRECRAWL_API_KEY."
            )

        self.client: Firecrawl = client or Firecrawl(api_key=api_key)

        # Agent is documented with FirecrawlApp; keep it optional so your code works even if not installed.
        if app is not None:
            self.app = app
        elif FirecrawlApp is not None and api_key:
            self.app = FirecrawlApp(api_key=api_key)  # type: ignore
        else:
            self.app = None

    # ============================================================
    # Core conversion helper
    # ============================================================

    def _to_dict(self, x: Any) -> Any:
        """
        Convert Firecrawl SDK return types to plain Python objects.
        """
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

    # ============================================================
    # Scrape payload normalization (handles SDK/version response shapes)
    # ============================================================

    def _extract_scrape_payload(self, raw: Any) -> Dict[str, Any]:
        """
        Normalize Firecrawl scrape responses to a stable shape:

          {
            "data": {...formats...} | None,
            "metadata": {...} | None,
            "raw": <original dict-safe payload>
          }

        Some SDK/version combos return:
          A) {"success": bool, "data": {...}, "metadata": {...}}
        Others surface formats at the top-level:
          B) {"markdown": "...", "metadata": {...}, ...}

        This is *not* site-specific; it's only response-shape normalization.
        """
        raw_obj = raw if isinstance(raw, dict) else {"value": raw}

        # Preferred: "data" holds the formats
        data_obj = raw_obj.get("data")
        metadata_obj = raw_obj.get("metadata")

        if isinstance(data_obj, dict):
            return {
                "data": data_obj,
                "metadata": metadata_obj if isinstance(metadata_obj, dict) else None,
                "raw": raw_obj,
            }

        # Fallback: formats at top-level
        known_format_keys = (
            "markdown",
            "html",
            "rawHtml",
            "raw_html",
            "links",
            "images",
            "screenshot",
            "json",
            "summary",
            "actions",
            "warning",
            "change_tracking",
            "branding",
        )

        extracted: Dict[str, Any] = {}
        for k in known_format_keys:
            if k in raw_obj and raw_obj.get(k) is not None:
                extracted[k] = raw_obj.get(k)

        return {
            "data": extracted or None,
            "metadata": metadata_obj if isinstance(metadata_obj, dict) else None,
            "raw": raw_obj,
        }

    # ============================================================
    # TWO CANONICAL "AGENT TOOLS"
    # ============================================================

    def search_tool(
        self,
        query: str,
        *,
        limit: int = 5,
        include_content: bool = False,
        content_formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        timeout_ms: int = 60000,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        AGENT TOOL #1: google-like search.

        If include_content=True, passes scrape_options so Firecrawl will scrape each search result and
        return content (markdown/html/etc.) along with url/title/description.

        Returns:
          normalize=True: {"query","limit","results":[...],"raw":...}
          normalize=False: dict-safe raw payload
        """
        if not query or not query.strip():
            return ToolResult.fail("search_tool: query is empty")

        try:
            scrape_options = None
            if include_content:
                if content_formats is None:
                    content_formats = ["markdown"]
                scrape_options = {
                    "formats": content_formats,
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

            results = self._normalize_search_results(raw_obj)
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
            return ToolResult.fail(f"Firecrawl search_tool failed for query={query!r}", e)

    def lookup_tool(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        timeout_ms: int = 120000,
        normalize: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        AGENT TOOL #2: Scrape a specific URL.
        Default: markdown only_main_content.

        NOTE:
        - This method is a primitive. It does NOT contain any site-specific logic.
        - It only normalizes Firecrawl's response shape so agents can reliably read `data["markdown"]`.
        """
        if not url or not url.strip():
            return ToolResult.fail("lookup_tool: url is empty")

        if formats is None:
            formats = ["markdown"]

        res = self.scrape(
            url,
            formats=formats,
            only_main_content=only_main_content,
            timeout_ms=timeout_ms,
            **kwargs,
        )
        if not res.ok or not normalize:
            return res

        payload = self._extract_scrape_payload(res.data)

        return ToolResult.success(
            {
                "url": url,
                "data": payload["data"],
                "metadata": payload["metadata"],
                "raw": payload["raw"],
            }
        )

    # ============================================================
    # Normalizers
    # ============================================================

    def _normalize_search_results(self, raw: Any) -> List[Dict[str, Any]]:
        """
        Best-effort normalization for Firecrawl Search.

        Output per item:
          {
            "title": str | None,
            "url": str | None,
            "description": str | None,
            "content": { "markdown": ..., "html": ..., ... } | None,
            "source": "firecrawl",
            "extra": {...}
          }
        """
        items: List[Any] = []

        if isinstance(raw, dict):
            # common shapes: {"web":[...]} or {"data":{"web":[...]}} etc.
            if isinstance(raw.get("web"), list):
                items = raw["web"]
            elif isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("web"), list):
                items = raw["data"]["web"]
            else:
                for k in ("results", "items"):
                    v = raw.get(k)
                    if isinstance(v, list):
                        items = v
                        break
        elif isinstance(raw, list):
            items = raw

        normalized: List[Dict[str, Any]] = []
        for it in items:
            it = it if isinstance(it, dict) else {"value": it}

            # v2 search results often include "metadata": {"title","url","description",...}
            md = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}

            url = (
                md.get("url")
                or it.get("url")
                or it.get("link")
                or it.get("href")
                or md.get("source_url")
            )
            title = md.get("title") or it.get("title") or it.get("name")
            desc = md.get("description") or it.get("description") or it.get("snippet") or it.get("summary")

            # Pull content fields if present
            content: Dict[str, Any] = {}
            for k in ("markdown", "html", "rawHtml", "raw_html", "links", "images", "screenshot", "json", "summary"):
                if k in it and it.get(k) is not None:
                    content[k] = it.get(k)

            # Extra: remove known keys to keep extra small-ish
            extra = dict(it)
            for k in (
                "metadata",
                "markdown",
                "html",
                "rawHtml",
                "raw_html",
                "links",
                "images",
                "screenshot",
                "json",
                "summary",
            ):
                extra.pop(k, None)

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

    # ============================================================
    # Scrape / Crawl / Map / Batch / Raw Search
    # ============================================================

    def scrape(
        self,
        url: str,
        *,
        formats: Optional[List[str]] = None,
        only_main_content: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Scrape a single URL.
        """
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
            return ToolResult.fail(f"Firecrawl scrape failed for url={url!r}", e)

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

            data = self.client.crawl(**params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl crawl failed for url={url!r}", e)

    def start_crawl(self, url: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        try:
            params: Dict[str, Any] = {"url": url}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            data = self.client.start_crawl(**params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl start_crawl failed for url={url!r}", e)

    def get_crawl_status(self, crawl_id: str, **kwargs: Any) -> ToolResult:
        try:
            data = self.client.get_crawl_status(crawl_id, **kwargs)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl get_crawl_status failed for id={crawl_id!r}", e)

    def cancel_crawl(self, crawl_id: str) -> ToolResult:
        try:
            data = self.client.cancel_crawl(crawl_id)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl cancel_crawl failed for id={crawl_id!r}", e)

    def get_crawl_status_page(self, next_url: str) -> ToolResult:
        try:
            data = self.client.get_crawl_status_page(next_url)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail("Firecrawl get_crawl_status_page failed", e)

    def map(self, url: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        try:
            params: Dict[str, Any] = {"url": url}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            data = self.client.map(**params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl map failed for url={url!r}", e)

    def search(self, query: str, *, limit: Optional[int] = None, **kwargs: Any) -> ToolResult:
        """Raw Firecrawl search (kept for backward-compat)."""
        try:
            params: Dict[str, Any] = {"query": query}
            if limit is not None:
                params["limit"] = int(limit)
            params.update(kwargs)
            data = self.client.search(**params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl search failed for query={query!r}", e)

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

            data = self.client.batch_scrape(list(urls), **params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail("Firecrawl batch_scrape failed", e)

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

            data = self.client.start_batch_scrape(list(urls), **params)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail("Firecrawl start_batch_scrape failed", e)

    def get_batch_scrape_status(self, batch_id: str, **kwargs: Any) -> ToolResult:
        try:
            data = self.client.get_batch_scrape_status(batch_id, **kwargs)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl get_batch_scrape_status failed for id={batch_id!r}", e)

    def get_batch_scrape_status_page(self, next_url: str) -> ToolResult:
        try:
            data = self.client.get_batch_scrape_status_page(next_url)
            return ToolResult.success(self._to_dict(data))
        except Exception as e:
            return ToolResult.fail("Firecrawl get_batch_scrape_status_page failed", e)

    # ============================================================
    # Agent (FirecrawlApp-based, per docs)
    # ============================================================

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
        """
        Agent mode (search + navigate + extract).

        - schema is OPTIONAL.
        - If you want non-blocking control, we do start_agent + poll get_agent_status.
        """
        if not prompt or not prompt.strip():
            return ToolResult.fail("agent_tool: prompt is empty")
        if self.app is None:
            return ToolResult.fail(
                "agent_tool: FirecrawlApp is not available in this environment. "
                "Install/upgrade the Firecrawl SDK that includes FirecrawlApp, or pass app=... in the constructor."
            )

        try:
            # Start job
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
            job_id = job_obj.get("id") or job_obj.get("jobId") or job_obj.get("job_id")
            if not job_id:
                job_id = getattr(job, "id", None)

            if not job_id:
                return ToolResult.fail(
                    f"agent_tool: could not determine job id from start_agent response: {job_obj}"
                )

            # Poll status until completed/failed/timeout
            deadline = time.time() + float(timeout_s)
            last_status: Optional[Dict[str, Any]] = None

            while time.time() < deadline:
                status = self.app.get_agent_status(job_id)
                status_obj = self._to_dict(status)
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
                        f"Firecrawl agent failed: {status_obj.get('error') or 'unknown error'}",
                        None,
                    )

                time.sleep(float(poll_interval_s))

            return ToolResult.fail(
                f"agent_tool: timed out after {timeout_s}s. last_status={last_status}"
            )
        except Exception as e:
            return ToolResult.fail("Firecrawl agent_tool failed", e)

    # Backward-compatible wrappers if you still want them:

    def agent(
        self,
        *,
        prompt: str,
        urls: Optional[Sequence[str]] = None,
        schema: Optional[Union[Type[T], Dict[str, Any]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Direct agent() call (blocking) if your SDK supports it.
        """
        if self.app is None:
            return ToolResult.fail(
                "agent: FirecrawlApp is not available. Install/upgrade SDK or pass app=... in constructor."
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

            res = self.app.agent(**params)
            return ToolResult.success(self._to_dict(res))
        except Exception as e:
            return ToolResult.fail("Firecrawl agent failed", e)

    def start_agent(self, *, prompt: str, urls: Optional[Sequence[str]] = None, **kwargs: Any) -> ToolResult:
        if self.app is None:
            return ToolResult.fail(
                "start_agent: FirecrawlApp is not available. Install/upgrade SDK or pass app=... in constructor."
            )
        try:
            params: Dict[str, Any] = {"prompt": prompt}
            if urls is not None:
                params["urls"] = list(urls)
            params.update(kwargs)
            job = self.app.start_agent(**params)
            return ToolResult.success(self._to_dict(job))
        except Exception as e:
            return ToolResult.fail("Firecrawl start_agent failed", e)

    def get_agent_status(self, agent_id: str, **kwargs: Any) -> ToolResult:
        if self.app is None:
            return ToolResult.fail(
                "get_agent_status: FirecrawlApp is not available. Install/upgrade SDK or pass app=... in constructor."
            )
        try:
            status = self.app.get_agent_status(agent_id, **kwargs)
            return ToolResult.success(self._to_dict(status))
        except Exception as e:
            return ToolResult.fail(f"Firecrawl get_agent_status failed for id={agent_id!r}", e)
