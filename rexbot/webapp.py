from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .assistant import AssistantController, SessionAuditLog
from .cli import ACK_PHRASE, build_controller
from .llama3 import Llama3Client


@dataclass
class DashboardService:
    controller: AssistantController
    client: Llama3Client | None = None
    chat_messages: list[dict[str, Any]] = field(default_factory=list)

    def manifest(self) -> list[dict[str, Any]]:
        return self.controller.tool_runner.insane_tool_manifest()

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": definition.description,
                "required": definition.parameters.get("required", []),
                "properties": definition.parameters.get("properties", {}),
            }
            for name, definition in sorted(self.controller.tool_registry.items())
        ]

    def system_info(self) -> dict[str, Any]:
        return self.controller.tool_runner.system_info()

    def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return self.controller.execute_tool(tool_name, arguments)

    def chat_turn(self, message: str) -> dict[str, Any]:
        if self.client is None:
            return {
                "ok": False,
                "error": "No Ollama/Llama 3 client configured for chat.",
            }
        self.chat_messages.append({"role": "user", "content": message})
        reply = self.controller.run_llama3_turn(self.client, self.chat_messages)
        self.chat_messages.append({"role": "assistant", "content": reply})
        return {"ok": True, "reply": reply, "messages": self.chat_messages}

    def reset_chat(self) -> dict[str, Any]:
        self.chat_messages.clear()
        return {"ok": True, "messages": []}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rexbot dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", action="append", default=["."])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--ack", default="")
    parser.add_argument("--audit-log", default="logs/session_audit.json")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--enable-chat", action="store_true")
    parser.add_argument("--max-tool-round-trips", type=int, default=8)
    return parser


def build_dashboard_service(args: argparse.Namespace) -> DashboardService:
    if args.supervised and not args.ack:
        args.ack = ACK_PHRASE
    controller = build_controller(args)
    controller.audit_log = SessionAuditLog(Path(args.audit_log))
    client = None
    if args.enable_chat:
        client = Llama3Client(
            model=args.model,
            endpoint=args.endpoint,
            context_window=args.context_window,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    return DashboardService(controller=controller, client=client)


def make_handler(service: DashboardService) -> type[BaseHTTPRequestHandler]:
    ui_root = Path(__file__).with_name("ui")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            routes = {
                "/api/manifest": service.manifest,
                "/api/tools": service.tools,
                "/api/system-info": service.system_info,
            }
            if self.path in routes:
                self._send_json(routes[self.path]())
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            payload = self._read_json()
            if self.path == "/api/tool":
                try:
                    result = service.run_tool(payload["tool_name"], payload.get("arguments", {}))
                    self._send_json({"ok": True, "result": result})
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/chat":
                self._send_json(service.chat_turn(payload.get("message", "")))
                return
            if self.path == "/api/reset-chat":
                self._send_json(service.reset_chat())
                return
            self._send_json({"ok": False, "error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)

        def _serve_static(self) -> None:
            target = self.path.strip("/") or "index.html"
            file_path = (ui_root / target).resolve()
            if ui_root.resolve() not in file_path.parents and file_path != ui_root.resolve() / "index.html":
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not file_path.exists() or file_path.is_dir():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = "text/html; charset=utf-8"
            if file_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            body = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(raw or "{}")

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = build_dashboard_service(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Rexbot dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
