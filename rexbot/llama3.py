from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from .tools import ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Llama3Response:
    content: str
    tool_calls: list[ToolCall]


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    approved: bool
    content: str


class Llama3Client:
    """Minimal Ollama-compatible client for local Llama 3 chat + tools."""

    def __init__(
        self,
        model: str = "llama3.1",
        endpoint: str = "http://localhost:11434/api/chat",
        system_prompt: str | None = None,
        context_window: int = 4096,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.system_prompt = system_prompt or (
            "You are Rexbot, a sharp, efficient Llama 3 assistant. "
            "Use the smallest tool needed, avoid wasting tokens, summarize clearly, "
            "and adapt your behavior to low-power devices when possible."
        )
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: dict[str, ToolDefinition],
    ) -> Llama3Response:
        raw = self._post_chat(
            messages=self._with_system_prompt(messages),
            tools=[tool.as_llama_tool() for tool in tools.values()],
            max_tokens=self.max_tokens,
            system_prompt=None,
        )
        return self._parse_response(raw)

    def review_plan(
        self,
        messages: list[dict[str, Any]],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ReviewDecision:
        review_prompt = (
            "You are Rexbot's execution reviewer. A tool action was proposed. "
            "Reply with CONFIRM or REJECT on the first word, then a short reason. "
            "Only confirm if the action still matches the user's goal and looks safe."
        )
        review_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Review this planned tool action before execution:\n"
                    f"tool={tool_name}\narguments={json.dumps(arguments, sort_keys=True)}"
                ),
            },
        ]
        raw = self._post_chat(
            messages=self._with_system_prompt(review_messages, override_system_prompt=review_prompt),
            tools=[],
            max_tokens=96,
            system_prompt=review_prompt,
        )
        content = raw.get("message", {}).get("content", "").strip()
        normalized = content.upper()
        approved = normalized.startswith("CONFIRM")
        return ReviewDecision(approved=approved, content=content)

    def _post_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "tools": tools,
            "options": {
                "num_ctx": self.context_window,
                "num_predict": max_tokens,
                "temperature": self.temperature,
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_response(self, raw: dict[str, Any]) -> Llama3Response:
        message = raw.get("message", {})
        content = message.get("content", "")
        parsed_calls = []
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            parsed_calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=function.get("arguments", {}) or {},
                )
            )
        return Llama3Response(content=content, tool_calls=parsed_calls)

    def _with_system_prompt(
        self,
        messages: list[dict[str, Any]],
        override_system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        system_prompt = override_system_prompt or self.system_prompt
        if messages and messages[0].get("role") == "system":
            return messages
        return [{"role": "system", "content": system_prompt}, *messages]
