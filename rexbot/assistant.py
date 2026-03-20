from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .llama3 import Llama3Client, ReviewDecision, ToolCall
from .policy import Action, PolicyDecision, PolicyEngine, SessionMode
from .tools import ToolDefinition, ToolRunner

ApprovalCallback = Callable[[str, dict[str, Any], PolicyDecision], bool]
PATH_ARGUMENT_TOOLS = {
    "list_dir": ("path",),
    "read_file": ("path",),
    "write_file": ("path",),
    "append_file": ("path",),
    "make_dir": ("path",),
    "move_file": ("src", "dest"),
    "copy_file": ("src", "dest"),
    "rename_file": ("path",),
    "delete_file_safe": ("path",),
    "organize_directory": ("path",),
    "file_info": ("path",),
    "directory_tree": ("path",),
    "glob_search": ("path",),
    "grep_text": ("path",),
    "add_knowledge": ("path",),
}
NETWORK_ARGUMENT_TOOLS = {"fetch_url": "url"}
DEFAULT_PATH_ARGUMENTS = {
    ("organize_directory", "path"): "~/Downloads",
}


class KillSwitchTriggered(RuntimeError):
    pass


@dataclass(slots=True)
class SessionAuditLog:
    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self.entries.append(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))


@dataclass(slots=True)
class StopController:
    """Explicit stop command controller for interactive sessions.

    This replaces the old Ctrl+C behavior so copy/paste shortcuts do not
    accidentally terminate the assistant. The local operator must type the exact
    stop command on its own line.
    """

    stop_command: str = "/kill"
    exit_commands: tuple[str, ...] = ("/exit", "/quit")

    def inspect_input(self, user_input: str) -> str | None:
        normalized = user_input.strip()
        if normalized == self.stop_command:
            raise KillSwitchTriggered("Kill switch triggered with /kill.")
        if normalized in self.exit_commands:
            return "exit"
        return None


@dataclass(slots=True)
class AssistantController:
    policy: PolicyEngine
    tool_runner: ToolRunner = field(default_factory=ToolRunner)
    audit_log: SessionAuditLog = field(
        default_factory=lambda: SessionAuditLog(Path("logs/session_audit.json"))
    )
    max_tool_round_trips: int = 8
    tool_registry: dict[str, ToolDefinition] = field(init=False)

    def __post_init__(self) -> None:
        self.tool_registry = self.tool_runner.build_registry()

    def enable_supervised_mode(self, confirmation_phrase: str) -> None:
        required = "I understand supervised mode still keeps scope limits"
        if confirmation_phrase.strip() != required:
            raise ValueError(
                "Supervised mode requires the exact acknowledgement phrase."
            )
        self.policy.set_mode(SessionMode.SUPERVISED)
        self.audit_log.record("mode_changed", mode=self.policy.mode.value)

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        approval_callback: ApprovalCallback | None = None,
    ) -> Any:
        if tool_name not in self.tool_registry:
            raise PermissionError(f"Unknown tool requested: {tool_name}")

        normalized_targets = self._normalize_targets(tool_name, arguments)
        decision = self._evaluate_targets(tool_name, normalized_targets)
        self.audit_log.record(
            "tool_requested",
            tool=tool_name,
            arguments=arguments,
            target=normalized_targets[0],
            targets=normalized_targets,
            decision=decision.reason,
            needs_confirmation=decision.needs_confirmation,
            allowed=decision.allowed,
        )
        self._enforce(tool_name, arguments, decision, approval_callback)

        result = self.tool_registry[tool_name].handler(**arguments)
        self.audit_log.record("tool_completed", tool=tool_name, targets=normalized_targets)
        return result

    def run_llama3_turn(
        self,
        client: Llama3Client,
        messages: list[dict[str, Any]],
        approval_callback: ApprovalCallback | None = None,
    ) -> str:
        transcript = list(messages)
        final_content = ""
        for _ in range(self.max_tool_round_trips):
            response = client.chat(transcript, self.tool_registry)
            if response.content:
                final_content = response.content
            if not response.tool_calls:
                self.audit_log.record("assistant_responded", content=final_content)
                return final_content

            self.audit_log.record(
                "tool_calls_requested",
                tools=[call.name for call in response.tool_calls],
            )
            rejected_plan = False
            for call in response.tool_calls:
                review = self._review_tool_plan(client, transcript, call)
                self.audit_log.record(
                    "tool_reviewed",
                    tool=call.name,
                    approved=review.approved,
                    review_message=review.content,
                )
                if not review.approved:
                    transcript.append(
                        {
                            "role": "system",
                            "content": (
                                f"Planned tool {call.name} with arguments {json.dumps(call.arguments, sort_keys=True)} "
                                f"was rejected during confirmation: {review.content}. Choose a better action."
                            ),
                        }
                    )
                    rejected_plan = True
                    break
                tool_result = self.execute_tool(call.name, call.arguments, approval_callback)
                transcript.append(self._assistant_tool_call_message(call))
                transcript.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": self._stringify_tool_result(tool_result),
                    }
                )
            if rejected_plan:
                continue
        raise RuntimeError("Llama 3 exceeded the maximum tool round trips.")

    def _review_tool_plan(
        self,
        client: Llama3Client,
        transcript: list[dict[str, Any]],
        tool_call: ToolCall,
    ) -> ReviewDecision:
        return client.review_plan(transcript, tool_call.name, tool_call.arguments)

    def _normalize_targets(self, tool_name: str, arguments: dict[str, Any]) -> list[str]:
        if tool_name in NETWORK_ARGUMENT_TOOLS:
            raw = str(arguments.get(NETWORK_ARGUMENT_TOOLS[tool_name], ""))
            parsed = urlparse(raw if "://" in raw else f"https://{raw}")
            return [parsed.netloc or parsed.path]
        if tool_name in PATH_ARGUMENT_TOOLS:
            return [
                str(
                    Path(
                        str(arguments.get(key, DEFAULT_PATH_ARGUMENTS.get((tool_name, key), "")))
                    ).expanduser().resolve()
                )
                for key in PATH_ARGUMENT_TOOLS[tool_name]
            ]
        return [tool_name]

    def _evaluate_targets(self, tool_name: str, normalized_targets: list[str]) -> PolicyDecision:
        decisions = [
            self.policy.evaluate(Action(kind=tool_name, target=target))
            for target in normalized_targets
        ]
        if any(not decision.allowed for decision in decisions):
            blocked = next(decision for decision in decisions if not decision.allowed)
            return blocked
        if any(decision.needs_confirmation for decision in decisions):
            return PolicyDecision(True, True, "Safe mode requires explicit confirmation.")
        return PolicyDecision(True, False, "All requested targets are inside approved scope.")

    def _enforce(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        decision: PolicyDecision,
        approval_callback: ApprovalCallback | None,
    ) -> None:
        if not decision.allowed:
            raise PermissionError(decision.reason)
        if decision.needs_confirmation:
            if approval_callback is None or not approval_callback(tool_name, arguments, decision):
                raise PermissionError(
                    f"Action requires interactive approval: {decision.reason}"
                )

    def _assistant_tool_call_message(self, tool_call: ToolCall) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                }
            ],
        }

    def _stringify_tool_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, indent=2)
