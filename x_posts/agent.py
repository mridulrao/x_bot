# agent.py
"""
Conversation-first ReACT X agent using OpenAI tool calling + your tools.

Key upgrades vs previous version:
- "Conversation mode" loop: if the user gives only an idea, the agent asks 1–3 targeted questions.
- Enforces explicit POST confirmation before any post_tweet tool call.
- Optionally enforces "must preview before posting" at code-level (recommended).

Integrates:
- search_tool.execute_tool: web_search, web_fetch
- x_tool.execute_tool: preview_tweet, post_tweet, check_auth_status
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

import search_tool
import x_tool
from prompt import SYSTEM_PROMPT


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")


# ---------------------------
# Config
# ---------------------------

@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.4
    max_output_tokens: int = 900

    max_steps: int = 12
    max_tool_calls: int = 30

    require_post_confirmation: bool = True
    enforce_preview_before_post: bool = True

    # Local testing: force posting tool into dry_run=True
    dry_run_posting: bool = False


# ---------------------------
# Tool schema conversion
# ---------------------------

def anthropic_tools_to_openai(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in anthropic_tools
    ]


OPENAI_TOOL_SPECS: List[Dict[str, Any]] = anthropic_tools_to_openai(
    (search_tool.TOOL_DEFINITIONS or []) + (x_tool.TOOL_DEFINITIONS or [])
)


# ---------------------------
# Tool dispatcher
# ---------------------------

POST_TOOL_NAMES = {"post_tweet"}
PREVIEW_TOOL_NAMES = {"preview_tweet"}


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s


def _tool_owned_by(module_tools: List[Dict[str, Any]], name: str) -> bool:
    return any(t.get("name") == name for t in (module_tools or []))


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if _tool_owned_by(search_tool.TOOL_DEFINITIONS, name):
        return search_tool.execute_tool(name, args)
    if _tool_owned_by(x_tool.TOOL_DEFINITIONS, name):
        return x_tool.execute_tool(name, args)
    return {"ok": False, "error": {"type": "UnknownTool", "message": f"Tool not found: {name}"}}


def user_confirmed_post(user_text: str) -> bool:
    return user_text.strip().lower() == "post"


# ---------------------------
# Agent
# ---------------------------

@dataclass
class ReActXAgent:
    client: OpenAI
    config: AgentConfig = field(default_factory=AgentConfig)

    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_used: int = 0

    post_confirmed: bool = False

    # Track whether we previewed since last edit
    last_preview_fingerprint: Optional[str] = None
    last_draft_fingerprint: Optional[str] = None

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_calls_used = 0
        self.post_confirmed = False
        self.last_preview_fingerprint = None
        self.last_draft_fingerprint = None

    def add_user(self, text: str) -> None:
        if user_confirmed_post(text):
            self.post_confirmed = True
        else:
            # Any non-POST user message could imply edits/changes; we keep confirmation but require preview again.
            pass
        self.messages.append({"role": "user", "content": text})

    def _chat(self) -> Any:
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=self.messages,
            tools=OPENAI_TOOL_SPECS if OPENAI_TOOL_SPECS else None,
            tool_choice="auto" if OPENAI_TOOL_SPECS else None,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )

    def _append_tool_result(self, tool_call_id: str, name: str, payload: Dict[str, Any]) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )

    def _fingerprint_text(self, content: str, hashtags: str = "") -> str:
        # Simple stable fingerprint for "draft content + hashtags"
        blob = (content or "").strip() + "\n#\n" + (hashtags or "").strip()
        return str(hash(blob))

    def step(self) -> Tuple[bool, str]:
        resp = self._chat()
        msg = resp.choices[0].message

        assistant_text = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None)

        # Save assistant message
        assistant_entry: Dict[str, Any] = {"role": "assistant"}
        if assistant_text:
            assistant_entry["content"] = assistant_text
        if tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        self.messages.append(assistant_entry)

        if not tool_calls:
            return True, assistant_text

        for tc in tool_calls:
            if self.tool_calls_used >= self.config.max_tool_calls:
                self._append_tool_result(
                    tc.id,
                    tc.function.name,
                    {"ok": False, "error": {"type": "ToolCallLimit", "message": "Max tool calls reached."}},
                )
                continue

            tool_name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            parsed = _safe_json_loads(raw_args)
            args: Dict[str, Any] = parsed if isinstance(parsed, dict) else {"_raw": parsed}

            # Track preview vs post content to enforce preview-before-post
            if tool_name in PREVIEW_TOOL_NAMES:
                content = str(args.get("content", ""))
                hashtags = str(args.get("hashtags", ""))
                self.last_draft_fingerprint = self._fingerprint_text(content, hashtags)

            # Posting gate: require user confirmation
            if tool_name in POST_TOOL_NAMES and self.config.require_post_confirmation and not self.post_confirmed:
                self._append_tool_result(
                    tc.id,
                    tool_name,
                    {
                        "ok": False,
                        "blocked": True,
                        "error": {
                            "type": "NeedsUserConfirmation",
                            "message": "User has not confirmed posting yet. Ask them to reply exactly 'POST' to proceed.",
                        },
                        "draft_received": args,
                    },
                )
                self.tool_calls_used += 1
                continue

            # Enforce preview before post (recommended)
            if tool_name in POST_TOOL_NAMES and self.config.enforce_preview_before_post:
                content = str(args.get("content", ""))
                hashtags = str(args.get("hashtags", ""))
                fp = self._fingerprint_text(content, hashtags)
                if self.last_preview_fingerprint != fp:
                    self._append_tool_result(
                        tc.id,
                        tool_name,
                        {
                            "ok": False,
                            "blocked": True,
                            "error": {
                                "type": "PreviewRequired",
                                "message": "Must run preview_tweet for the exact same content/hashtags immediately before posting.",
                            },
                            "hint": "Call preview_tweet with the same content/hashtags, then retry post_tweet.",
                        },
                    )
                    self.tool_calls_used += 1
                    continue

            # Optional: force dry_run for posting
            if tool_name in POST_TOOL_NAMES and self.config.dry_run_posting:
                args = dict(args)
                args["dry_run"] = True

            t0 = time.time()
            result = call_tool(tool_name, args)
            dt_ms = int((time.time() - t0) * 1000)

            # If preview succeeded, set preview fingerprint
            if tool_name in PREVIEW_TOOL_NAMES:
                content = str(args.get("content", ""))
                hashtags = str(args.get("hashtags", ""))
                self.last_preview_fingerprint = self._fingerprint_text(content, hashtags)

            payload = {"ok": True, "tool": tool_name, "ms": dt_ms, "result": result}
            self._append_tool_result(tc.id, tool_name, payload)
            self.tool_calls_used += 1

        return False, assistant_text

    def run(self, user_input: str) -> str:
        if not self.messages:
            self.reset()

        self.add_user(user_input)

        final_text = ""
        for _ in range(self.config.max_steps):
            done, txt = self.step()
            if txt:
                final_text = txt
            if done:
                return final_text

        # Fallback
        self.messages.append(
            {"role": "user", "content": "Stop tool use now. Summarize status and ask the next single question."}
        )
        resp = self._chat()
        out = (resp.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": out})
        return out


# ---------------------------
# CLI loop (conversation)
# ---------------------------

def main() -> int:
    if not OPENAI_API_KEY:
        print("Missing OPENAI_API_KEY env var.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=OPENAI_API_KEY)
    agent = ReActXAgent(client=client, config=AgentConfig())
    agent.reset()

    print("Conversation-first ReACT X Agent ready.")
    print("Type your idea. The agent will ask 1–3 questions, draft options, refine, then ask you to type POST.\n")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        out = agent.run(user_text)
        if out:
            print(f"\nagent> {out}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())