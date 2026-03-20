from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assistant import (
    AssistantController,
    KillSwitchTriggered,
    SessionAuditLog,
    StopController,
)
from .llama3 import Llama3Client
from .policy import PolicyDecision, PolicyEngine


ACK_PHRASE = "I understand supervised mode still keeps scope limits"
LEGACY_TARGET_TOOLS = {
    "list_dir",
    "read_file",
    "write_file",
    "append_file",
    "make_dir",
    "file_info",
    "directory_tree",
    "glob_search",
    "grep_text",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama 3-driven local assistant prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tool_parser = subparsers.add_parser("tool", help="Run one tool directly")
    tool_parser.add_argument("tool_name")
    tool_parser.add_argument("target", nargs="?")
    tool_parser.add_argument("--content", default="")
    tool_parser.add_argument("--arg", action="append", default=[])
    tool_parser.add_argument("--args-json", default="")
    _add_shared_flags(tool_parser)

    chat_parser = subparsers.add_parser("chat", help="Run an interactive Llama 3 tool-using chat")
    chat_parser.add_argument("--model", default="llama3.1")
    chat_parser.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    chat_parser.add_argument(
        "--system-prompt",
        default=(
            "You are Rexbot, a smarter Llama 3 assistant that uses tools carefully. "
            "Choose tools automatically when they help, be efficient, low-resource friendly, clear, and deliberate. "
            "If the user asks to organize downloads without a path, you may use organize_directory with its default Downloads target."
        ),
    )
    chat_parser.add_argument("--context-window", type=int, default=4096)
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.2)
    chat_parser.add_argument("--max-tool-round-trips", type=int, default=8)
    _add_shared_flags(chat_parser)
    return parser


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", action="append", default=["."])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument(
        "--ack",
        default="",
        help=f"Required exact phrase for supervised mode: {ACK_PHRASE}",
    )
    parser.add_argument("--audit-log", default="logs/session_audit.json")


def build_controller(args: argparse.Namespace) -> AssistantController:
    policy = PolicyEngine(
        allowed_roots=[Path(root) for root in args.root],
        allowed_domains=set(args.domain),
        allow_network=args.allow_network,
    )
    controller = AssistantController(
        policy=policy,
        audit_log=SessionAuditLog(Path(args.audit_log)),
        max_tool_round_trips=getattr(args, "max_tool_round_trips", 8),
    )
    if args.supervised:
        controller.enable_supervised_mode(args.ack)
    return controller


def prompt_for_approval(
    tool_name: str,
    arguments: dict[str, Any],
    decision: PolicyDecision,
) -> bool:
    print(
        f"Approve tool '{tool_name}' with arguments {arguments}? "
        f"[{decision.reason}] (y/N): ",
        end="",
        flush=True,
    )
    return input().strip().lower() in {"y", "yes"}


def parse_tool_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.args_json:
        return json.loads(args.args_json)

    payload: dict[str, Any] = {}
    for pair in args.arg:
        key, value = pair.split("=", 1)
        payload[key] = _coerce_cli_value(value)

    if args.target:
        payload.setdefault("url" if args.tool_name == "fetch_url" else "path", args.target)
    if args.content:
        payload.setdefault("content", args.content)
    if payload:
        return payload

    if args.tool_name in LEGACY_TARGET_TOOLS and args.target:
        return {"path": args.target}
    if args.tool_name == "fetch_url" and args.target:
        return {"url": args.target}
    return {}


def _coerce_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def run_tool_command(args: argparse.Namespace) -> int:
    controller = build_controller(args)
    try:
        payload = parse_tool_payload(args)
        result = controller.execute_tool(
            args.tool_name,
            payload,
            None if args.supervised else prompt_for_approval,
        )
        print(result)
        return 0
    except Exception as exc:
        controller.audit_log.record("action_failed", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1


def run_chat_command(args: argparse.Namespace) -> int:
    controller = build_controller(args)
    stop_controller = StopController()
    client = Llama3Client(
        model=args.model,
        endpoint=args.endpoint,
        system_prompt=args.system_prompt,
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    messages: list[dict[str, Any]] = []
    print("Rexbot chat started. Type /kill to stop immediately or /exit to leave the session.")

    try:
        while True:
            user_input = input("you> ")
            command = stop_controller.inspect_input(user_input)
            if command == "exit":
                controller.audit_log.record("session_exited")
                return 0
            messages.append({"role": "user", "content": user_input})
            reply = controller.run_llama3_turn(
                client,
                messages,
                None if args.supervised else prompt_for_approval,
            )
            messages.append({"role": "assistant", "content": reply})
            print(f"rexbot> {reply}")
    except KillSwitchTriggered as exc:
        controller.audit_log.record("kill_switch_triggered", reason=str(exc))
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:
        controller.audit_log.record("action_failed", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tool":
        return run_tool_command(args)
    return run_chat_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
