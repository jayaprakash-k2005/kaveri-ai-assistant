"""Reusable streaming Ollama client for the Midstone 4 assignment."""
from __future__ import annotations

import json
import os
import random
from email.utils import parsedate_to_datetime
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 5
BASE_WAIT_S = 1.0
MAX_WAIT_S = 30.0


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)   # [{id, name, arguments(dict)}]
    model: str = ""
    streamed: bool = False
    chunks: int = 0
    latency_ms: int = 0


class RunLog:
    """Appends one JSON object per event to a .jsonl file (or does nothing)."""

    def __init__(self, path: str | None = None):
        self.path = path

    def write(self, **event) -> None:
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class LLMClient:
    def __init__(self, log: RunLog | None = None, max_retries: int = MAX_RETRIES):
        self.base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL")
        # max_retries=0: the openai package must not retry silently; we retry ourselves.
        self._client = openai.OpenAI(base_url=self.base_url, api_key=os.getenv("LLM_API_KEY"),
                                     max_retries=0, timeout=float(os.getenv("LLM_TIMEOUT", "300")))
        self.create = self._client.chat.completions.create   # replaceable in tests
        self.log = log or RunLog()
        self.max_retries = min(MAX_RETRIES, max(0, max_retries))

    def _call(self, **kwargs):
        """One API call, retried on HTTP 429 with a growing wait."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.create(model=self.model, **kwargs)
            except openai.RateLimitError as err:
                if attempt == self.max_retries:
                    raise
                retry_after = None
                response = getattr(err, "response", None)
                headers = getattr(response, "headers", {}) or {}
                value = headers.get("retry-after")
                if value is not None:
                    try:
                        retry_after = max(0.0, float(value))
                    except (TypeError, ValueError):
                        try:
                            retry_after = max(0.0, (parsedate_to_datetime(value).timestamp() - time.time()))
                        except (TypeError, ValueError, OverflowError):
                            retry_after = None
                if retry_after is None:
                    wait = min(MAX_WAIT_S, BASE_WAIT_S * (2 ** attempt) + random.uniform(0.0, 0.25))
                else:
                    wait = min(MAX_WAIT_S, retry_after)
                self.log.write(event="retry", status=429, attempt=attempt + 1, wait_s=round(wait, 2))
                time.sleep(wait)
            # Any other error is NOT caught here: retrying a wrong key only hides it.

    def chat(self, messages: list[dict], tools: list[dict] | None = None, stream: bool = True,
             on_token: Callable[[str], None] | None = None) -> Reply:
        started = time.perf_counter()
        extra = {"tools": tools} if tools else {}
        reply = Reply(streamed=stream, model=self.model)
        if not stream:
            resp = self._call(messages=messages, **extra)
            msg = resp.choices[0].message
            reply.content = msg.content or ""
            reply.tool_calls = [_tool_call(tc.id, tc.function.name, tc.function.arguments)
                                for tc in (msg.tool_calls or [])]
        else:
            pieces: dict[int, dict] = {}   # tool calls arrive in pieces, keyed by index
            text = []
            for chunk in self._call(messages=messages, stream=True, **extra):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    reply.chunks += 1
                    text.append(delta.content)
                    if on_token:
                        on_token(delta.content)
                for tc in delta.tool_calls or []:
                    slot = pieces.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] += tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            reply.content = "".join(text)
            reply.tool_calls = [_tool_call(p["id"], p["name"], p["arguments"])
                                for _i, p in sorted(pieces.items())]
        reply.latency_ms = int((time.perf_counter() - started) * 1000)
        return reply


def _tool_call(call_id: str, name: str, arguments: str | dict) -> dict:
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    return {"id": call_id or f"call_{uuid.uuid4().hex[:8]}", "name": name, "arguments": arguments}


class Conversation:
    """Chat history plus tool calling. One instance per conversation."""

    def __init__(self, client: LLMClient, system: str | None = None, tools: list[dict] | None = None,
                 functions: dict[str, Callable[..., Any]] | None = None, conversation_id: str = "c1"):
        self.client = client
        self.history: list[dict] = [{"role": "system", "content": system}] if system else []
        self.tools = tools
        self.functions = functions or {}
        self.id = conversation_id
        self.turn = 0

    def ask(self, text: str, on_token: Callable[[str], None] | None = None) -> str:
        self.turn += 1
        log = self.client.log
        self.history.append({"role": "user", "content": text})
        log.write(conversation_id=self.id, turn=self.turn, role="user", content=text)

        for _round in range(4):
            reply = self.client.chat(self.history, tools=self.tools, on_token=on_token)
            event = dict(conversation_id=self.id, turn=self.turn, role="assistant", content=reply.content,
                         streamed=reply.streamed, chunks=reply.chunks, model=reply.model,
                         latency_ms=reply.latency_ms)
            if not reply.tool_calls:
                self.history.append({"role": "assistant", "content": reply.content})
                log.write(**event)
                return reply.content

            log.write(**event, tool_calls=reply.tool_calls)
            # The assistant's tool request goes into the history in the API's own shape.
            self.history.append({"role": "assistant", "content": reply.content or None, "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in reply.tool_calls]})
            for tc in reply.tool_calls:
                name = tc["name"]
                if name not in self.functions:
                    raise ValueError(f"Model requested unknown tool: {name}")
                result = str(self.functions[name](**tc["arguments"]))
                self.history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                log.write(conversation_id=self.id, turn=self.turn, role="tool",
                          tool_call_id=tc["id"], name=name, content=result)
        raise RuntimeError("The model kept calling tools without answering.")


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """An OpenAI-style function tool definition."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}
