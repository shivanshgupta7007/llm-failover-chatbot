"""LLM chatbot backend: LiteLLM key-pool failover, response caching, SQLite sessions."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import litellm
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuration (all through environment variables)
# ---------------------------------------------------------------------------


def _csv(name: str, default: str = "") -> list[str]:
    return [p.strip() for p in os.environ.get(name, default).split(",") if p.strip()]


API_KEYS = _csv("LLM_API_KEYS")
KEY_IDS = [f"key_{i + 1}" for i in range(len(API_KEYS))]
MODEL = os.environ.get("LLM_MODEL", "gemini/gemini-3.1-flash-lite")
FALLBACK_MODELS = _csv("LLM_FALLBACK_MODELS", "gemini/gemini-3.1-flash-lite,gemini/gemini-3.8-flash,gemini/gemini-3.6-flash")
MODEL_CHAIN = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:////data/app.db")
ALLOWED_ORIGINS = _csv("ALLOWED_ORIGINS", "http://localhost:3000")
FAILURE_SIMULATION = os.environ.get("ENABLE_FAILURE_SIMULATION", "false").lower() == "true"
CACHE_DIR = os.environ.get("LITELLM_CACHE_DIR", "/data/litellm-cache")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "45"))
LLM_NUM_RETRIES = int(os.environ.get("LLM_NUM_RETRIES", "2"))
LLM_CHAIN_PASSES = int(os.environ.get("LLM_CHAIN_PASSES", "3"))
LLM_RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "3"))
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT", "You are a concise, helpful assistant."
)


def _db_path(url: str) -> str:
    prefix = "sqlite:///"
    return url[len(prefix):] if url.startswith(prefix) else url


DB_PATH = _db_path(DATABASE_URL)
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# LiteLLM cache (disk-backed when possible, in-memory fallback)
# ---------------------------------------------------------------------------

CACHE_ACTIVE = False
if os.environ.get("DISABLE_CACHE", "false").lower() != "true":
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        litellm.cache = litellm.Cache(type="disk", disk_cache_dir=CACHE_DIR)
        CACHE_ACTIVE = True
    except Exception:
        try:
            litellm.cache = litellm.Cache(type="local")
            CACHE_ACTIVE = True
        except Exception:
            CACHE_ACTIVE = False

# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row


def init_db() -> None:
    with _lock, _conn:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                system_prompt TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
        )
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_usage (
                cache_key TEXT PRIMARY KEY,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL
            )
            """
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_message(session_id: str, role: str, content: str, metadata: Optional[dict]) -> None:
    with _lock, _conn:
        _conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content,
             json.dumps(metadata) if metadata is not None else None, _now()),
        )


def load_history(session_id: str) -> list[dict]:
    with _lock:
        rows = _conn.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[sqlite3.Row]:
    with _lock:
        return _conn.execute(
            "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()


def upsert_session(session_id: str, system_prompt: Optional[str]) -> None:
    with _lock, _conn:
        _conn.execute(
            "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?)"
            " ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt",
            (session_id, system_prompt, _now()),
        )


def remember_usage(cache_key: str, input_tokens: int, output_tokens: int) -> None:
    with _lock, _conn:
        _conn.execute(
            "INSERT OR IGNORE INTO cache_usage (cache_key, input_tokens, output_tokens)"
            " VALUES (?, ?, ?)",
            (cache_key, input_tokens, output_tokens),
        )


def recall_usage(cache_key: str) -> Optional[tuple[int, int]]:
    with _lock:
        row = _conn.execute(
            "SELECT input_tokens, output_tokens FROM cache_usage WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    return (row["input_tokens"], row["output_tokens"]) if row else None


init_db()

# ---------------------------------------------------------------------------
# Failure-simulation hook
# ---------------------------------------------------------------------------

SIM_STATUSES = {429, 500, 502, 503}
_KEY_RE = re.compile(r"^(key_\d+|\*)$")


def parse_simulation(header: Optional[str]) -> Optional[dict[str, int]]:
    """Parse X-Simulate-Failure into {key_id: status}; None when disabled/absent."""
    if not FAILURE_SIMULATION or not header:
        return None
    out: dict[str, int] = {}
    for pair in header.split(","):
        key, sep, val = pair.partition("=")
        if not sep:
            continue
        key = key.strip()
        try:
            status = int(val.strip())
        except ValueError:
            continue
        if status in SIM_STATUSES and _KEY_RE.match(key):
            out[key] = status
    return out or None


# ---------------------------------------------------------------------------
# LLM call with key-pool failover
# ---------------------------------------------------------------------------


class AllProvidersUnavailable(Exception):
    pass


def _cache_key(model: str, messages: list[dict]) -> str:
    blob = json.dumps([model] + [[m["role"], m["content"]] for m in messages],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def llm_chat(messages: list[dict], simulation: Optional[dict[str, int]]) -> dict:
    """Try each key in order, then fall back across models. Simulated keys fail
    without a network call.

    Cache reads are bypassed while a simulation header is active so failover is
    demonstrated against live keys, exactly as the header contract describes.
    """
    attempts = 0
    errors: list[str] = []
    use_cache = CACHE_ACTIVE and simulation is None
    # Demand spikes on free tiers last seconds to minutes; ride them out with a
    # few passes over the whole chain. Simulated failures get a single pass so
    # the failover contract answers immediately.
    passes = 1 if simulation is not None else LLM_CHAIN_PASSES

    for pass_no in range(passes):
      if pass_no:
        time.sleep(LLM_RETRY_DELAY)
      for model in MODEL_CHAIN:
        ck = _cache_key(model, messages)
        for kid, api_key in zip(KEY_IDS, API_KEYS):
            sim_status = simulation.get(kid, simulation.get("*")) if simulation else None
            if sim_status is not None:
                attempts += 1
                errors.append(f"{kid}: simulated HTTP {sim_status}")
                continue
            try:
                resp = litellm.completion(
                    model=model, api_key=api_key, messages=messages, caching=use_cache,
                    timeout=LLM_TIMEOUT, num_retries=LLM_NUM_RETRIES,
                )
            except Exception as exc:  # 429, quota, 5xx, auth, network -> next key
                attempts += 1
                errors.append(f"{kid}@{model}: {type(exc).__name__}")
                continue

            hidden = getattr(resp, "_hidden_params", None) or {}
            cache_hit = bool(hidden.get("cache_hit"))
            text = ""
            try:
                text = (resp.choices[0].message.content or "").strip()
            except Exception:
                text = ""
            if not text:
                attempts += 1
                errors.append(f"{kid}@{model}: empty completion")
                continue

            usage = getattr(resp, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            if cache_hit and not (input_tokens or output_tokens):
                stored = recall_usage(ck)
                if stored:
                    input_tokens, output_tokens = stored
            elif not cache_hit and (input_tokens or output_tokens):
                remember_usage(ck, input_tokens, output_tokens)

            return {
                "reply": text,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_hit": cache_hit,
                "key_id": None if cache_hit else kid,
                "fallback_attempts": 0 if cache_hit else attempts,
            }

    raise AllProvidersUnavailable("; ".join(errors) or "no keys configured")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("session_id", "message", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


app = FastAPI(title="LLM chatbot with key failover and response caching")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health")
def health():
    try:
        with _lock:
            _conn.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False
    status = "ok" if db_ok and API_KEYS else "degraded"
    return {
        "status": status,
        "keys_configured": len(API_KEYS),
        "cache_enabled": CACHE_ACTIVE,
    }


@app.post("/chat")
def chat(req: ChatRequest, x_simulate_failure: Optional[str] = Header(default=None)):
    simulation = parse_simulation(x_simulate_failure)

    session = get_session(req.session_id)
    if req.system_prompt:
        system_prompt = req.system_prompt
        upsert_session(req.session_id, system_prompt)
    elif session and session["system_prompt"]:
        system_prompt = session["system_prompt"]
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    history = load_history(req.session_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages += [{"role": r["role"], "content": r["content"]} for r in history]
    messages.append({"role": "user", "content": req.message})

    started = time.perf_counter()
    try:
        result = llm_chat(messages, simulation)
    except AllProvidersUnavailable:
        return JSONResponse(
            status_code=503, content={"detail": "all providers unavailable"}
        )
    latency_ms = int((time.perf_counter() - started) * 1000)

    metadata = {
        "resolution_source": "LLM_CACHE" if result["cache_hit"] else "LIVE_INFERENCE",
        "model": result["model"],
        "latency_ms": latency_ms,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "key_id": result["key_id"],
        "fallback_attempts": result["fallback_attempts"],
    }

    if session is None and not req.system_prompt:
        upsert_session(req.session_id, None)
    save_message(req.session_id, "user", req.message, None)
    save_message(req.session_id, "assistant", result["reply"], metadata)

    return {"session_id": req.session_id, "reply": result["reply"], "metadata": metadata}


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    if get_session(session_id) is None:
        return JSONResponse(status_code=404, content={"detail": "session not found"})
    out = []
    for row in load_history(session_id):
        item = {"role": row["role"], "content": row["content"]}
        if row["role"] == "assistant" and row["metadata"]:
            item["metadata"] = json.loads(row["metadata"])
        out.append(item)
    return {"session_id": session_id, "messages": out}
