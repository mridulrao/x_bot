"""
x_tools.py

LLM-compatible tool definitions wrapping XPoster.
Provides:
  - TOOL_DEFINITIONS  : list of Anthropic-style tool schemas to pass to the API
  - execute_tool()    : dispatcher — call this with (tool_name, tool_input) from
                        the model's tool_use block

Usage:
    from x_tools import TOOL_DEFINITIONS, execute_tool

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
from pathlib import Path
from typing import Any, Dict, Optional

from integrations.x import XPoster, default_token_path
from integrations.x import get_user_access_token

from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------ #
#  Singleton poster — shared across all tool calls in a session       #
# ------------------------------------------------------------------ #

_poster: Optional[XPoster] = None


def _get_poster() -> XPoster:
    global _poster
    if _poster is None:
        _poster = XPoster()
    return _poster


# ------------------------------------------------------------------ #
#  Tool schemas (Anthropic tool-use format)                           #
# ------------------------------------------------------------------ #

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "post_tweet",
        "description": (
            "Post a single tweet or a thread of tweets to X (Twitter). "
            "Long content is automatically split at sentence boundaries so no tweet "
            "exceeds 280 characters. Hashtags are normalized and appended to the last "
            "tweet in the thread. Set dry_run=true to preview without actually posting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The main text to post. May be longer than 280 chars; it will be split into a thread automatically.",
                },
                "hashtags": {
                    "type": "string",
                    "description": (
                        "Optional hashtags as a comma- or space-separated string, e.g. 'ai, ml, #agents'. "
                        "The # prefix is optional. Max 8 tags."
                    ),
                    "default": "",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, validate and preview the tweet(s) without posting to X. Default false.",
                    "default": False,
                },
                "create_thread": {
                    "type": "boolean",
                    "description": "If true (default), multi-tweet content is posted as a reply thread. If false, only the first chunk is posted.",
                    "default": True,
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "preview_tweet",
        "description": (
            "Preview how a tweet (or thread) would look after splitting and hashtag "
            "normalization, without posting anything. Equivalent to post_tweet with dry_run=true. "
            "Returns the list of tweet chunks and any warnings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The text you want to preview.",
                },
                "hashtags": {
                    "type": "string",
                    "description": "Optional hashtags string, e.g. 'ai, ml, #agents'.",
                    "default": "",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "check_auth_status",
        "description": (
            "Check whether a valid X OAuth2 token exists and is not expired. "
            "Returns ok=true if the poster is ready to post, otherwise an error message "
            "explaining what the user needs to do (e.g. run --auth)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ------------------------------------------------------------------ #
#  Individual tool handlers                                           #
# ------------------------------------------------------------------ #

def _handle_post_tweet(inp: Dict[str, Any]) -> Dict[str, Any]:
    content: str = inp.get("content", "")
    hashtags: str = inp.get("hashtags", "")
    dry_run: bool = bool(inp.get("dry_run", False))
    create_thread: bool = bool(inp.get("create_thread", True))

    result = _get_poster().post(
        content=content,
        hashtags=hashtags,
        dry_run=dry_run,
        create_thread=create_thread,
    )

    return _serialize_result(result)


def _handle_preview_tweet(inp: Dict[str, Any]) -> Dict[str, Any]:
    content: str = inp.get("content", "")
    hashtags: str = inp.get("hashtags", "")

    result = _get_poster().post(
        content=content,
        hashtags=hashtags,
        dry_run=True,           # always dry-run for preview
        create_thread=True,
    )

    return _serialize_result(result)


def _handle_check_auth_status(_inp: Dict[str, Any]) -> Dict[str, Any]:

    token_path = default_token_path()
    access, err = get_user_access_token(token_path)

    if err:
        return {
            "ok": False,
            "error": err,
            "token_path": str(token_path),
            "action_required": "Run `python x_post.py --auth` to authenticate.",
        }

    return {
        "ok": True,
        "message": "Token is valid and ready to use.",
        "token_path": str(token_path),
    }


# ------------------------------------------------------------------ #
#  Result serialiser (PostResult → plain dict for the LLM)           #
# ------------------------------------------------------------------ #

def _serialize_result(result) -> Dict[str, Any]:
    """Convert a PostResult dataclass into a JSON-serialisable dict."""
    out: Dict[str, Any] = {
        "ok": result.ok,
        "dry_run": result.dry_run,
    }

    if not result.ok:
        out["error_code"] = result.error_code
        out["error_message"] = result.error_message

    if result.warnings:
        out["warnings"] = result.warnings

    if result.tweets:
        out["tweets"] = [
            {"index": i + 1, "total": len(result.tweets), "char_count": t.char_count, "text": t.text}
            for i, t in enumerate(result.tweets)
        ]

    if result.tweet_ids:
        out["tweet_ids"] = result.tweet_ids
        out["thread_url"] = f"https://x.com/i/web/status/{result.tweet_ids[0]}"

    return out


# ------------------------------------------------------------------ #
#  Public dispatcher                                                  #
# ------------------------------------------------------------------ #

_HANDLERS = {
    "post_tweet": _handle_post_tweet,
    "preview_tweet": _handle_preview_tweet,
    "check_auth_status": _handle_check_auth_status,
}


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a tool_use block from the LLM.

    Args:
        tool_name:  The `name` field from the model's tool_use block.
        tool_input: The `input` dict from the model's tool_use block.

    Returns:
        A plain dict that can be JSON-serialised and sent back as a
        tool_result content block.
    """
    handler = _HANDLERS.get(tool_name)
    if handler is None:
        return {
            "ok": False,
            "error_code": "unknown_tool",
            "error_message": f"No handler registered for tool '{tool_name}'. Available: {list(_HANDLERS)}",
        }

    try:
        return handler(tool_input)
    except Exception as exc:
        return {
            "ok": False,
            "error_code": "tool_exception",
            "error_message": str(exc),
        }


# ------------------------------------------------------------------ #
#  Quick smoke-test (python x_tools.py)                              #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== preview_tweet smoke test ===\n")
    out = execute_tool(
        "preview_tweet",
        {
            "content": (
                "Just shipped a new voice agent with real-time Hindi support using Whisper and Azure Speech. "
                "The latency is finally under 400ms end-to-end. "
                "Next up: multi-language switching mid-call."
            ),
            "hashtags": "ai, voiceagents, #livekit, ml",
        },
    )
    print(json.dumps(out, indent=2))

    print("\n=== check_auth_status ===\n")
    out2 = execute_tool("check_auth_status", {})
    print(json.dumps(out2, indent=2))