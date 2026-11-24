"""Backend contract tests. LiteLLM is faked; cache semantics are emulated."""
import json
import os
import sys
import types

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["LLM_API_KEYS"] = "test-key-1,test-key-2"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DIR}/test.db"
os.environ["LITELLM_CACHE_DIR"] = f"{TEST_DIR}/test-cache"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"
os.environ["ENABLE_FAILURE_SIMULATION"] = "true"

sys.path.insert(0, os.path.dirname(TEST_DIR))

import litellm  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402


def _fake_response(text, model, in_tok, out_tok, cache_hit=False):
    msg = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=msg)
    usage = types.SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok)
    return types.SimpleNamespace(
        choices=[choice], usage=usage, model=model,
        _hidden_params={"cache_hit": cache_hit},
    )


@pytest.fixture(autouse=True)
def fresh_db():
    with main._lock, main._conn:
        main._conn.execute("DROP TABLE IF EXISTS messages")
        main._conn.execute("DROP TABLE IF EXISTS sessions")
        main._conn.execute("DROP TABLE IF EXISTS cache_usage")
    main.init_db()
    yield


@pytest.fixture
def client():
    return TestClient(main.app, raise_server_exceptions=False)


def fake_completion_factory(store):
    """Fake litellm.completion with litellm-style cache semantics."""
    def fake_completion(model, api_key, messages, caching=True, **kwargs):
        blob = json.dumps([model] + [[m["role"], m["content"]] for m in messages])
        if caching and blob in store:
            cached = store[blob]
            return _fake_response(cached["text"], model, cached["in"], cached["out"],
                                  cache_hit=True)
        if api_key == "bad-key":
            raise litellm.exceptions.APIError(
                status_code=429, message="rate limited",
                llm_provider="fake", model=model)
        text = f"answer:{messages[-1]['content']}"
        store[blob] = {"text": text, "in": 10 + sum(len(m["content"]) for m in messages), "out": 5}
        return _fake_response(text, model, store[blob]["in"], store[blob]["out"])
    return fake_completion


@pytest.fixture
def fake_llm(monkeypatch):
    store = {}
    monkeypatch.setattr(main.litellm, "completion", fake_completion_factory(store))
    return store


# --- health ---------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


# --- validation ------------------------------------------------------------

def test_empty_message_422(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "   "})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_long_message_422(client):
    r = client.post("/chat", json={"session_id": "s1", "message": "x" * 4001})
    assert r.status_code == 422


def test_missing_session_id_422(client):
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 422


def test_malformed_json_422(client):
    r = client.post("/chat", content="{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# --- chat + metadata --------------------------------------------------------

def test_chat_success_metadata(client, fake_llm):
    r = client.post("/chat", json={"session_id": "s-a", "message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s-a"
    assert body["reply"] == "answer:hello"
    md = body["metadata"]
    assert md["resolution_source"] == "LIVE_INFERENCE"
    assert md["model"]
    assert isinstance(md["latency_ms"], int)
    assert md["input_tokens"] > 0 and md["output_tokens"] > 0
    assert md["key_id"] == "key_1"
    assert md["fallback_attempts"] == 0


def test_cache_hit_second_session(client, fake_llm):
    r1 = client.post("/chat", json={"session_id": "s-1", "message": "capital of France?"})
    r2 = client.post("/chat", json={"session_id": "s-2", "message": "capital of France?"})
    assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    md2 = r2.json()["metadata"]
    assert md2["resolution_source"] == "LLM_CACHE"
    assert md2["key_id"] is None
    assert md2["fallback_attempts"] == 0
    assert r2.json()["reply"] == r1.json()["reply"]
    assert md2["input_tokens"] == r1.json()["metadata"]["input_tokens"]
    assert md2["latency_ms"] <= r1.json()["metadata"]["latency_ms"]


def test_per_request_tokens_match_across_sessions(client, fake_llm):
    r1 = client.post("/chat", json={"session_id": "t-1", "message": "same question"})
    r2 = client.post("/chat", json={"session_id": "t-2", "message": "same question"})
    assert (r1.json()["metadata"]["input_tokens"]
            == r2.json()["metadata"]["input_tokens"])


# --- failover ---------------------------------------------------------------

def test_failover_to_second_key(client, fake_llm):
    r = client.post("/chat", json={"session_id": "f-1", "message": "q"},
                    headers={"X-Simulate-Failure": "key_1=429"})
    assert r.status_code == 200
    md = r.json()["metadata"]
    assert md["key_id"] == "key_2"
    assert md["fallback_attempts"] == 1
    assert md["resolution_source"] == "LIVE_INFERENCE"


def test_all_keys_fail_503_nothing_stored(client, fake_llm):
    r = client.post("/chat", json={"session_id": "f-2", "message": "q"},
                    headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 503
    assert r.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/f-2/messages").status_code == 404


def test_simulation_ignored_when_flag_off(client, fake_llm, monkeypatch):
    monkeypatch.setattr(main, "FAILURE_SIMULATION", False)
    r = client.post("/chat", json={"session_id": "f-3", "message": "q"},
                    headers={"X-Simulate-Failure": "*=429"})
    assert r.status_code == 200


# --- sessions / history -----------------------------------------------------

def test_multi_turn_history(client, fake_llm):
    client.post("/chat", json={"session_id": "m-1", "message": "first"})
    client.post("/chat", json={"session_id": "m-1", "message": "second"})
    r = client.get("/sessions/m-1/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["metadata"]["resolution_source"] in ("LIVE_INFERENCE", "LLM_CACHE")
    # history is sent to the model on turn 2
    sent = list(fake_llm.keys())[-1]
    assert "first" in sent and "second" in sent


def test_sessions_are_separate(client, fake_llm):
    client.post("/chat", json={"session_id": "x-1", "message": "only in x1"})
    r = client.get("/sessions/x-2/messages")
    assert r.status_code == 404
    assert r.json() == {"detail": "session not found"}


def test_unknown_session_404(client):
    assert client.get("/sessions/nope/messages").status_code == 404


def test_system_prompt_override(client, fake_llm):
    client.post("/chat", json={"session_id": "p-1", "message": "hi",
                               "system_prompt": "be terse"})
    sent = list(fake_llm.keys())[0]
    assert "be terse" in sent
    # history endpoint never exposes the system prompt
    msgs = client.get("/sessions/p-1/messages").json()["messages"]
    assert all(m["role"] in ("user", "assistant") for m in msgs)
