# LLM Chatbot with Key Failover and Response Caching

A small full-stack AI chat app: React frontend, FastAPI backend, LiteLLM gateway
with an API-key pool, automatic failover, response caching, and SQLite session
storage.

## 1. Live URLs

- Frontend: https://llm-failover-chatbot-web.onrender.com
- Backend: https://llm-failover-chatbot-api.onrender.com (`GET /health` shows key pool + cache state)

## 2. Run locally

### With Docker

```bash
cp .env.example .env       # fill in LLM_API_KEYS (at least two keys)
make up                    # backend on :8000, frontend on :3000
make down
```

### Without Docker

Backend:

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
LLM_API_KEYS="keyA,keyB" DATABASE_URL="sqlite:////tmp/app.db" \
  ENABLE_FAILURE_SIMULATION=true uvicorn app.main:app --port 8000
```

Frontend:

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Backend tests:

```bash
cd backend && . .venv/bin/activate && python -m pytest tests/ -q
```

## 3. Design decisions

### Key fallback

`LLM_API_KEYS` is a comma-separated pool; ids are `key_1`, `key_2`, ... in
order. Every request walks the pool in order. A key that returns HTTP 429, a
quota error, a 5xx, an auth error or a network/timeout failure is skipped and
the next key is tried, inside the same request. If every key fails on the
primary model, the same pool is retried against `LLM_FALLBACK_MODELS` (the
contract allows keys across one or more models). Only if everything fails does
the client get `503 {"detail": "all providers unavailable"}`, and nothing is
written to the session history. The response metadata reports which key
answered (`key_id`) and how many keys failed before it (`fallback_attempts`),
never the key material itself.

`POST /chat` also honors the `X-Simulate-Failure` header (e.g.
`key_1=429`, `key_1=429,key_2=500`, `*=429`) when
`ENABLE_FAILURE_SIMULATION=true`. Simulated keys fail without a network call,
and cache reads are bypassed for that request so the failover is demonstrated
against live keys and `*=429` always yields the 503 contract response.

### Caching

LiteLLM's built-in cache is enabled at startup (disk-backed under
`LITELLM_CACHE_DIR`, with an in-memory fallback). The cache key is the model
plus the full message list (system prompt + history + new message), so the
same first message in a new session is served from cache as `LLM_CACHE` with
the identical reply text, `key_id: null` and `fallback_attempts: 0`, while any
different message is a live call. Token usage is stored with the cached
response (with a SQLite side table as belt-and-braces) so cached replies still
report the original per-request token counts. Cache hits skip the network
round trip, so they are measurably faster than live calls.

### SQLite session storage

All state is in one SQLite file whose path comes from `DATABASE_URL`
(default `sqlite:////data/app.db`). In Docker, `/data` is a named volume, so
history survives container restarts. The schema is created automatically on
startup: the required `messages` table, plus a `sessions` table (per-session
system prompt) and a `cache_usage` table (token usage for cached replies).
The client only sends the new message; the backend rebuilds the full message
list (system prompt + stored history + new message) for every call. Failed
requests (422/503) are never written to history.

## 4. Verification

- Failover: `curl -X POST $BACKEND/chat -H 'Content-Type: application/json' \
  -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"demo-1","message":"hello"}'` returns 200 with
  `"key_id": "key_2"` and `"fallback_attempts": 1`. With `*=429` it returns
  `503 {"detail": "all providers unavailable"}` and stores nothing.
- Cache hit: send the same first message in two different sessions (or reload
  the page and repeat a question). The second reply shows `LLM_CACHE`, the
  same text, `key_id: null` and a much lower latency.
- Per-message tokens: every assistant bubble in the UI shows `input_tokens`
  and `output_tokens` for that request only; two identical first messages in
  two sessions report the same input tokens.
- History: `GET /sessions/{session_id}/messages` returns the stored history
  oldest-first from SQLite; an unknown session returns
  `404 {"detail": "session not found"}`.

## 5. Setup Used

- Editor: VS Code
- AI tools: an AI coding assistant (LLM pair-programmer) for scaffolding,
  tests and debugging; LiteLLM and FastAPI documentation.

## 6. Bonus features

- Model-level fallback chain (`LLM_FALLBACK_MODELS`) on top of the key pool,
  so a saturated model degrades to a secondary model instead of an error.
- Disk-backed LiteLLM cache (survives process restarts) with automatic
  in-memory fallback.
- Per-session system prompt stored server-side and editable from the UI.
- Backend test suite (15 tests) covering the API contract, failover hook,
  caching behavior, validation and multi-turn history.
