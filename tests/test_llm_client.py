import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from src.llm_client import Conversation, LLMClient, RunLog, tool_spec


@pytest.fixture(autouse=True)
def local_ollama_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")


def rate_limit(headers=None):
    response = httpx.Response(429, headers=headers or {}, request=httpx.Request("POST", "http://test"))
    return openai.RateLimitError("rate limited", response=response, body={})


def test_initialization(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_API_KEY", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    client = LLMClient()
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model == "qwen2.5:7b"
    assert client._client.max_retries == 0


def test_streaming_chunks_callback_and_tool_fragments(monkeypatch):
    client = LLMClient()
    chunks = [
        NS(choices=[NS(delta=NS(content="hello ", tool_calls=[]))]),
        NS(choices=[NS(delta=NS(content="world", tool_calls=[NS(index=0, id="call_", function=NS(name="get_", arguments='{"x":'))]))]),
        NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id="1", function=NS(name="revenue", arguments="1}"))]))]),
    ]
    client.create = lambda **kwargs: iter(chunks)
    got = []
    reply = client.chat([], tools=[{"type": "function"}], on_token=got.append)
    assert reply.content == "hello world"
    assert reply.chunks == 2
    assert got == ["hello ", "world"]
    assert reply.streamed is True
    assert reply.tool_calls == [{"id": "call_1", "name": "get_revenue", "arguments": {"x": 1}}]


def test_multiple_tool_calls_grouped_by_index():
    client = LLMClient()
    calls = [
        NS(index=1, id="b", function=NS(name="two", arguments='{"n":2}')),
        NS(index=0, id="a", function=NS(name="one", arguments='{"n":1}')),
    ]
    client.create = lambda **kwargs: iter([NS(choices=[NS(delta=NS(content=None, tool_calls=calls))])])
    assert [tc["name"] for tc in client.chat([], tools=[{}]).tool_calls] == ["one", "two"]


def test_conversation_history_tool_execution_and_logging():
    log_path = Path(__file__).with_name("_test_run_log.jsonl")
    log_path.unlink(missing_ok=True)
    log = RunLog(str(log_path))
    client = LLMClient(log=log)
    replies = iter([
        NS(content="", tool_calls=[{"id": "call-1", "name": "get_revenue", "arguments": {"country": "Germany", "year": 2024}}],
           streamed=True, chunks=0, model="local", latency_ms=1),
        NS(content="Revenue was 90000.", tool_calls=[], streamed=True, chunks=3, model="local", latency_ms=2),
    ])
    seen = []
    def chat(messages, **kwargs):
        seen.append(list(messages))
        return next(replies)
    client.chat = chat
    convo = Conversation(client, system="sys", tools=[tool_spec("get_revenue", "", {}, [])],
                         functions={"get_revenue": lambda **kw: f"{kw['country']}={kw['year']}:90000"})
    assert convo.ask("Revenue?") == "Revenue was 90000."
    assert seen[1][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "Germany=2024:90000"}
    assert convo.history[-1] == {"role": "assistant", "content": "Revenue was 90000."}
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["role"] == "user"
    assert events[1]["tool_calls"][0]["id"] == "call-1"
    assert events[2]["tool_call_id"] == "call-1"
    assert events[3]["content"] == "Revenue was 90000."
    log_path.unlink()


@pytest.mark.parametrize("failures", [1, 2])
def test_retries_then_succeeds(monkeypatch, failures):
    client = LLMClient(max_retries=5)
    monkeypatch.setattr("src.llm_client.time.sleep", lambda _: None)
    calls = 0
    def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise rate_limit({"retry-after": "0"})
        return "ok"
    client.create = create
    assert client._call(messages=[]) == "ok"
    assert calls == failures + 1


def test_successful_third_attempt():
    client = LLMClient(max_retries=5)
    client.create = lambda **kwargs: "ok"
    assert client._call(messages=[]) == "ok"


def test_retry_exhaustion(monkeypatch):
    client = LLMClient(max_retries=2)
    monkeypatch.setattr("src.llm_client.time.sleep", lambda _: None)
    client.create = lambda **kwargs: (_ for _ in ()).throw(rate_limit({"retry-after": "99"}))
    with pytest.raises(openai.RateLimitError):
        client._call(messages=[])


def test_retry_count_is_capped_at_five():
    assert LLMClient(max_retries=99).max_retries == 5


def test_non_429_propagates():
    client = LLMClient()
    error = openai.APIStatusError("bad", response=httpx.Response(500, request=httpx.Request("POST", "http://test")), body={})
    client.create = lambda **kwargs: (_ for _ in ()).throw(error)
    with pytest.raises(openai.APIStatusError):
        client._call(messages=[])
