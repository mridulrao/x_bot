#!/usr/bin/env python3
"""
x_post.py

Agent-friendly X poster:
- OAuth2 user-context PKCE tokens (stored in ~/.config/x_posts/token.json by default)
- Validation + hashtag normalization
- Sentence-aware splitting into multiple tweets (thread)
- Returns structured pass/fail results for agent integration

Env:
  X_CLIENT_ID=...
  X_REDIRECT_URI=http://localhost:8080/callback  (must match portal)
  X_CLIENT_SECRET=...                            (optional; only for confidential clients)
  X_SCOPES="tweet.write tweet.read users.read offline.access"   (optional)
  X_TOKEN_PATH="~/.config/x_posts/token.json"                  (optional)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Config
# -----------------------------

DEFAULT_MAX_LEN = 280
DEFAULT_SCOPES = "tweet.write tweet.read users.read offline.access"

AUTH_URL = "https://twitter.com/i/oauth2/authorize"
TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
TWEET_POST_URL = "https://api.x.com/2/tweets"

HASHTAG_RE = re.compile(r"^[A-Za-z0-9_]+$")


# -----------------------------
# Results / Errors (agent-friendly)
# -----------------------------

@dataclass
class TweetChunk:
    text: str
    char_count: int


@dataclass
class PostResult:
    ok: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    tweets: List[TweetChunk] = field(default_factory=list)      # what was (or would be) posted
    tweet_ids: List[str] = field(default_factory=list)          # returned by API
    raw_responses: List[dict] = field(default_factory=list)     # per tweet response
    dry_run: bool = False


# -----------------------------
# Path helpers
# -----------------------------

def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def default_token_path() -> Path:
    return expand_path(os.getenv("X_TOKEN_PATH", "~/.config/x_posts/token.json"))


# -----------------------------
# Hashtag helpers
# -----------------------------

def _split_hashtags(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\n\r\t ]+", raw.strip())
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("#"):
            p = p[1:]
        out.append(p)
    return out


def normalize_hashtags(raw: str, *, dedupe: bool = True, max_tags: int = 8) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    tags = _split_hashtags(raw)

    cleaned: List[str] = []
    seen = set()
    for t in tags:
        t2 = t.strip()
        if not t2:
            continue

        if "-" in t2:
            warnings.append(f"Hashtag '{t2}' contains '-'. Replacing '-' -> '_'")
            t2 = t2.replace("-", "_")

        if not HASHTAG_RE.match(t2):
            warnings.append(f"Hashtag '{t2}' contains unsupported characters; dropping it")
            continue

        if dedupe:
            key = t2.lower()
            if key in seen:
                continue
            seen.add(key)

        cleaned.append(t2)

    if len(cleaned) > max_tags:
        warnings.append(f"Too many hashtags ({len(cleaned)}). Keeping first {max_tags}.")
        cleaned = cleaned[:max_tags]

    return [f"#{t}" for t in cleaned], warnings


def append_hashtags(text: str, hashtags: List[str]) -> str:
    if not hashtags:
        return text.strip()
    text = (text or "").rstrip()
    if not text:
        return " ".join(hashtags)
    return text + "\n\n" + " ".join(hashtags)


# -----------------------------
# Sentence splitting (no mid-sentence cuts)
# -----------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter:
    - Splits on ". " / "! " / "? "
    - Keeps punctuation
    Not perfect, but good enough and deterministic.
    """
    t = (text or "").strip()
    if not t:
        return []
    parts = _SENTENCE_SPLIT_RE.split(t)
    # Clean empty fragments
    return [p.strip() for p in parts if p.strip()]


def split_text_into_tweets(
    content: str,
    hashtags: List[str],
    *,
    max_len: int = DEFAULT_MAX_LEN,
    hashtags_on_last_tweet: bool = True,
) -> Tuple[List[TweetChunk], List[str], Optional[str]]:
    """
    Returns (chunks, warnings, error_message_if_unrecoverable)
    - Splits content by sentence boundaries.
    - Ensures each tweet <= max_len.
    - If hashtags_on_last_tweet=True, hashtags are appended only to the final tweet.
    """
    warnings: List[str] = []
    content = (content or "").strip()

    if not content and not hashtags:
        return [], warnings, "Empty content and no hashtags."

    sentences = split_into_sentences(content) if content else []

    # Special case: content empty, hashtags only
    if not sentences:
        final = append_hashtags("", hashtags)
        if len(final) > max_len:
            return [], warnings, f"Hashtags-only post exceeds {max_len} chars."
        return [TweetChunk(final, len(final))], warnings, None

    # If putting hashtags on last tweet, reserve space for them in the last chunk.
    hashtags_text = ""
    if hashtags and hashtags_on_last_tweet:
        hashtags_text = "\n\n" + " ".join(hashtags)

    chunks: List[str] = []
    cur = ""

    def can_add(candidate: str) -> bool:
        return len(candidate) <= max_len

    for s in sentences:
        # Build tentative with space if needed
        if not cur:
            tentative = s
        else:
            tentative = cur + " " + s

        if can_add(tentative):
            cur = tentative
            continue

        # Can't fit -> flush current
        if cur:
            chunks.append(cur)
            cur = s
            # If single sentence itself too long, we cannot split "without cutting sentences"
            if len(cur) > max_len:
                return [], warnings, (
                    f"A single sentence is longer than {max_len} chars, cannot split without cutting. "
                    f"Sentence starts: {cur[:80]!r}"
                )
        else:
            # cur empty but sentence too long
            if len(s) > max_len:
                return [], warnings, (
                    f"A single sentence is longer than {max_len} chars, cannot split without cutting. "
                    f"Sentence starts: {s[:80]!r}"
                )
            cur = s

    if cur:
        chunks.append(cur)

    # Now attach hashtags
    if hashtags:
        if hashtags_on_last_tweet:
            last = chunks[-1] + hashtags_text
            if len(last) <= max_len:
                chunks[-1] = last
            else:
                # If hashtags don't fit on last tweet, try:
                # 1) put hashtags as their own tweet
                tag_only = " ".join(hashtags)
                if len(tag_only) <= max_len:
                    warnings.append("Hashtags did not fit in last tweet; posting as final hashtags-only tweet.")
                    chunks.append(tag_only)
                else:
                    return [], warnings, f"Hashtags exceed {max_len} chars even alone."
        else:
            # Append hashtags to every tweet (not recommended; can blow char budget)
            new_chunks = []
            for c in chunks:
                cc = append_hashtags(c, hashtags)
                if len(cc) > max_len:
                    return [], warnings, "Hashtags-on-every-tweet mode overflowed character limit."
                new_chunks.append(cc)
            chunks = new_chunks

    tweet_chunks = [TweetChunk(text=c, char_count=len(c)) for c in chunks]
    return tweet_chunks, warnings, None


# -----------------------------
# OAuth2 PKCE (same as your working version)
# -----------------------------

def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def make_code_verifier() -> str:
    return b64url(secrets.token_bytes(32))


def make_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return b64url(digest)


class _CallbackHandler(BaseHTTPRequestHandler):
    code: Optional[str] = None
    error: Optional[str] = None
    state: Optional[str] = None
    expected_state: Optional[str] = None

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        q = parse_qs(urlparse(self.path).query)
        if "error" in q:
            _CallbackHandler.error = q["error"][0]
        if "state" in q:
            _CallbackHandler.state = q["state"][0]
        if "code" in q:
            _CallbackHandler.code = q["code"][0]

        ok = _CallbackHandler.code is not None and (
            _CallbackHandler.expected_state is None or _CallbackHandler.state == _CallbackHandler.expected_state
        )
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h3>Authorized. You can close this tab.</h3>" if ok else b"<h3>Authorization failed.</h3>")


def parse_host_port_from_redirect(redirect_uri: str) -> Tuple[str, int]:
    u = urlparse(redirect_uri)
    host = u.hostname or "127.0.0.1"
    port = u.port or (443 if u.scheme == "https" else 80)
    return host, port


def load_tokens(token_path: Path) -> Optional[Dict[str, Any]]:
    if not token_path.exists():
        return None
    try:
        return json.loads(token_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_tokens(token_path: Path, tokens: Dict[str, Any]) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")


def token_is_expired(tokens: Dict[str, Any]) -> bool:
    exp = int(tokens.get("_expires_at", 0) or 0)
    return exp == 0 or time.time() >= exp


def oauth2_authenticate_interactive(client_id: str, redirect_uri: str, scopes: str, token_path: Path) -> Dict[str, Any]:
    verifier = make_code_verifier()
    challenge = make_code_challenge(verifier)
    state = secrets.token_urlsafe(16)

    _CallbackHandler.code = None
    _CallbackHandler.error = None
    _CallbackHandler.state = None
    _CallbackHandler.expected_state = state

    host, port = parse_host_port_from_redirect(redirect_uri)
    httpd = HTTPServer((host, port), _CallbackHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    url = AUTH_URL + "?" + urlencode(params)

    print("\nOpen this URL if your browser does not open automatically:\n")
    print(url + "\n")
    webbrowser.open(url)

    while _CallbackHandler.code is None and _CallbackHandler.error is None:
        time.sleep(0.05)

    httpd.shutdown()

    if _CallbackHandler.error:
        raise RuntimeError(f"OAuth error: {_CallbackHandler.error}")
    if _CallbackHandler.state != state:
        raise RuntimeError("OAuth state mismatch (possible CSRF).")

    code = _CallbackHandler.code
    token_data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }

    client_secret = (os.getenv("X_CLIENT_SECRET") or "").strip()
    if client_secret:
        token_data["client_secret"] = client_secret

    r = requests.post(TOKEN_URL, data=token_data, timeout=30)
    tokens = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"Token exchange failed {r.status_code}: {json.dumps(tokens, ensure_ascii=False)}")

    expires_in = int(tokens.get("expires_in", 0) or 0)
    tokens["_obtained_at"] = int(time.time())
    tokens["_expires_at"] = int(time.time()) + max(0, expires_in - 30)

    save_tokens(token_path, tokens)
    print(f"\nSaved tokens to: {token_path}")
    return tokens


def refresh_access_token(client_id: str, refresh_token: str, token_path: Path) -> Dict[str, Any]:
    client_secret = (os.getenv("X_CLIENT_SECRET") or "").strip()
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret

    r = requests.post(TOKEN_URL, data=data, timeout=30)
    newt = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    if r.status_code >= 400:
        raise RuntimeError(f"Token refresh failed {r.status_code}: {json.dumps(newt, ensure_ascii=False)}")

    merged = dict(newt)
    if "refresh_token" not in merged:
        merged["refresh_token"] = refresh_token

    expires_in = int(merged.get("expires_in", 0) or 0)
    merged["_obtained_at"] = int(time.time())
    merged["_expires_at"] = int(time.time()) + max(0, expires_in - 30)

    save_tokens(token_path, merged)
    return merged


def get_user_access_token(token_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (access_token, error_message)
    """
    client_id = (os.getenv("X_CLIENT_ID") or "").strip()
    redirect_uri = (os.getenv("X_REDIRECT_URI") or "").strip()
    if not client_id:
        return None, "Missing X_CLIENT_ID in env."
    if not redirect_uri:
        return None, "Missing X_REDIRECT_URI in env."

    tokens = load_tokens(token_path)
    if not tokens:
        return None, f"No token file found at {token_path}. Run with --auth."

    if token_is_expired(tokens):
        rt = (tokens.get("refresh_token") or "").strip()
        if not rt:
            return None, "Access token expired and no refresh_token found. Run --auth again."
        try:
            tokens = refresh_access_token(client_id, rt, token_path)
        except Exception as e:
            return None, f"Token refresh failed: {e}"

    access = (tokens.get("access_token") or "").strip()
    if not access:
        return None, "No access_token in token file. Run --auth again."
    return access, None


# -----------------------------
# XPoster class (agent primitive)
# -----------------------------

class XPoster:
    def __init__(
        self,
        *,
        token_path: Optional[Path] = None,
        max_len: int = DEFAULT_MAX_LEN,
        hashtags_on_last_tweet: bool = True,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.token_path = token_path or default_token_path()
        self.max_len = max_len
        self.hashtags_on_last_tweet = hashtags_on_last_tweet
        self.http = session or requests.Session()

    def ensure_auth_interactive(self) -> PostResult:
        """
        One-time interactive auth helper (useful for setup scripts, not agents).
        """
        client_id = (os.getenv("X_CLIENT_ID") or "").strip()
        redirect_uri = (os.getenv("X_REDIRECT_URI") or "").strip()
        scopes = (os.getenv("X_SCOPES") or DEFAULT_SCOPES).strip()

        if not client_id or not redirect_uri:
            return PostResult(
                ok=False,
                error_code="missing_env",
                error_message="X_CLIENT_ID and X_REDIRECT_URI must be set in env.",
            )

        try:
            oauth2_authenticate_interactive(client_id, redirect_uri, scopes, self.token_path)
            return PostResult(ok=True)
        except Exception as e:
            return PostResult(ok=False, error_code="auth_failed", error_message=str(e))

    def _post_one(self, text: str, access_token: str, reply_to_tweet_id: Optional[str]) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
        """
        Returns (tweet_id, raw_json, error_message)
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"text": text}
        if reply_to_tweet_id:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to_tweet_id}

        resp = self.http.post(TWEET_POST_URL, headers=headers, json=payload, timeout=30)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}

        if resp.status_code >= 400:
            return None, data, f"X API error {resp.status_code}: {json.dumps(data, ensure_ascii=False)}"

        tweet_id = (data.get("data") or {}).get("id")
        if not tweet_id:
            return None, data, "Posted but no tweet id returned in response."
        return str(tweet_id), data, None

    def post(
        self,
        *,
        content: str,
        hashtags: str = "",
        dry_run: bool = False,
        create_thread: bool = True,
    ) -> PostResult:
        """
        Agent-facing method.
        - Validates + normalizes hashtags
        - Splits into sentence-aligned tweet chunks if needed
        - Posts (optionally as a thread)
        """
        warnings: List[str] = []
        content = (content or "").strip()

        normalized_tags, tag_warnings = normalize_hashtags(hashtags)
        warnings.extend(tag_warnings)

        chunks, split_warnings, split_error = split_text_into_tweets(
            content=content,
            hashtags=normalized_tags,
            max_len=self.max_len,
            hashtags_on_last_tweet=self.hashtags_on_last_tweet,
        )
        warnings.extend(split_warnings)

        if split_error:
            return PostResult(ok=False, error_code="split_failed", error_message=split_error, warnings=warnings)

        if not chunks:
            return PostResult(ok=False, error_code="empty", error_message="Nothing to post after processing.", warnings=warnings)

        # Dry run: return chunks without calling X
        if dry_run:
            return PostResult(ok=True, warnings=warnings, tweets=chunks, dry_run=True)

        access, err = get_user_access_token(self.token_path)
        if err:
            return PostResult(ok=False, error_code="auth_missing", error_message=err, warnings=warnings, tweets=chunks)

        tweet_ids: List[str] = []
        raw_responses: List[dict] = []

        prev_id: Optional[str] = None
        for i, chunk in enumerate(chunks):
            reply_to = prev_id if (create_thread and prev_id) else None
            tweet_id, raw, post_err = self._post_one(chunk.text, access, reply_to)
            if raw is not None:
                raw_responses.append(raw)
            if post_err:
                return PostResult(
                    ok=False,
                    error_code="post_failed",
                    error_message=f"Failed on tweet {i+1}/{len(chunks)}: {post_err}",
                    warnings=warnings,
                    tweets=chunks,
                    tweet_ids=tweet_ids,
                    raw_responses=raw_responses,
                    dry_run=False,
                )
            tweet_ids.append(tweet_id)  # type: ignore[arg-type]
            prev_id = tweet_id

        return PostResult(
            ok=True,
            warnings=warnings,
            tweets=chunks,
            tweet_ids=tweet_ids,
            raw_responses=raw_responses,
            dry_run=False,
        )


# -----------------------------
# CLI wrapper (optional, still useful)
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent-friendly X poster (OAuth2 PKCE).")
    p.add_argument("--content", type=str, default="", help="Post content text.")
    p.add_argument("--content-file", type=str, default="", help="Path to file containing content.")
    p.add_argument("--hashtags", type=str, default="", help="Hashtags: 'ai,ml,#agents' or 'ai ml #agents'.")
    p.add_argument("--dry-run", action="store_true", help="Do not post; just print what would be posted.")
    p.add_argument("--no-thread", action="store_true", help="Do not create a thread; post only first chunk.")
    p.add_argument("--auth", action="store_true", help="Run one-time interactive OAuth auth and save tokens.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    poster = XPoster()

    if args.auth:
        res = poster.ensure_auth_interactive()
        if not res.ok:
            print(f"AUTH FAILED: {res.error_code}: {res.error_message}")
            return 10
        print("AUTH OK")
        return 0

    content = args.content
    if args.content_file:
        content = Path(args.content_file).read_text(encoding="utf-8")

    # If user disables thread, we still split (for preview), but only post first chunk
    res = poster.post(
        content=content,
        hashtags=args.hashtags,
        dry_run=args.dry_run,
        create_thread=not args.no_thread,
    )

    if not res.ok:
        print("\n--- FAIL ---")
        print(f"error_code: {res.error_code}")
        print(f"error_message: {res.error_message}")
        if res.warnings:
            print("warnings:")
            for w in res.warnings:
                print(f"  - {w}")
        return 2

    # Success
    print("\n--- OK ---")
    if res.warnings:
        print("warnings:")
        for w in res.warnings:
            print(f"  - {w}")

    print("\n--- TWEETS ---")
    for i, t in enumerate(res.tweets, 1):
        print(f"\n[{i}/{len(res.tweets)}] ({t.char_count}/{DEFAULT_MAX_LEN})")
        print(t.text)

    if not res.dry_run:
        print("\n--- POSTED IDS ---")
        print(res.tweet_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
