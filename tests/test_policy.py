from pathlib import Path

import pytest

from rexbot.assistant import AssistantController, SessionAuditLog, StopController
from rexbot.llama3 import Llama3Response, ReviewDecision, ToolCall
from rexbot.policy import Action, PolicyEngine, SessionMode
from rexbot.tools import ToolRunner
from rexbot.webapp import DashboardService


class StubLlama3Client:
    def __init__(
        self,
        responses: list[Llama3Response],
        reviews: list[ReviewDecision] | None = None,
    ) -> None:
        self.responses = responses
        self.reviews = reviews or []
        self.calls = 0
        self.review_calls = 0

    def chat(self, messages, tools):  # noqa: ANN001, ANN201
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def review_plan(self, messages, tool_name, arguments):  # noqa: ANN001, ANN201
        review = self.reviews[self.review_calls]
        self.review_calls += 1
        return review


def test_safe_mode_requires_confirmation(tmp_path: Path) -> None:
    policy = PolicyEngine(allowed_roots=[tmp_path])
    decision = policy.evaluate(Action("read_file", str(tmp_path / "notes.txt")))
    assert decision.allowed is True
    assert decision.needs_confirmation is True


def test_supervised_mode_allows_scoped_file_action(tmp_path: Path) -> None:
    policy = PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED)
    decision = policy.evaluate(Action("write_file", str(tmp_path / "notes.txt")))
    assert decision.allowed is True
    assert decision.needs_confirmation is False


def test_path_outside_scope_is_blocked(tmp_path: Path) -> None:
    policy = PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED)
    decision = policy.evaluate(Action("read_file", str(tmp_path.parent / "secret.txt")))
    assert decision.allowed is False


def test_meta_tool_is_allowed_without_prompt(tmp_path: Path) -> None:
    policy = PolicyEngine(allowed_roots=[tmp_path])
    decision = policy.evaluate(Action("system_info", "system_info"))
    assert decision.allowed is True
    assert decision.needs_confirmation is False


def test_memory_tool_is_allowed_without_prompt(tmp_path: Path) -> None:
    policy = PolicyEngine(allowed_roots=[tmp_path])
    decision = policy.evaluate(Action("save_memory", "save_memory"))
    assert decision.allowed is True
    assert decision.needs_confirmation is False


def test_controller_refuses_missing_supervised_phrase(tmp_path: Path) -> None:
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path]),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    with pytest.raises(ValueError):
        controller.enable_supervised_mode("yes")


def test_controller_executes_supervised_write(tmp_path: Path) -> None:
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    result = controller.execute_tool(
        "write_file",
        {"path": str(tmp_path / "notes.txt"), "content": "hello"},
    )
    assert "wrote" in result
    assert (tmp_path / "notes.txt").read_text() == "hello"


def test_controller_blocks_move_when_destination_outside_scope(tmp_path: Path) -> None:
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    source = tmp_path / "notes.txt"
    source.write_text("hello")
    with pytest.raises(PermissionError):
        controller.execute_tool(
            "move_file",
            {"src": str(source), "dest": str(tmp_path.parent / "notes.txt")},
        )


def test_stop_controller_uses_explicit_commands() -> None:
    controller = StopController()
    assert controller.inspect_input("copy text") is None
    assert controller.inspect_input("/exit") == "exit"
    with pytest.raises(RuntimeError):
        controller.inspect_input("/kill")


def test_llama3_turn_executes_tool_calls_after_review_confirmation(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello from llama3")
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    stub = StubLlama3Client(
        responses=[
            Llama3Response(content="", tool_calls=[ToolCall("read_file", {"path": str(target)})]),
            Llama3Response(content="I read the file successfully.", tool_calls=[]),
        ],
        reviews=[ReviewDecision(approved=True, content="CONFIRM safe and on-task")],
    )

    result = controller.run_llama3_turn(stub, [{"role": "user", "content": "read it"}])

    assert result == "I read the file successfully."
    assert stub.review_calls == 1


def test_rejected_tool_plan_gets_revised(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello")
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    stub = StubLlama3Client(
        responses=[
            Llama3Response(content="", tool_calls=[ToolCall("read_file", {"path": str(target)})]),
            Llama3Response(content="I will avoid that and answer directly.", tool_calls=[]),
        ],
        reviews=[ReviewDecision(approved=False, content="REJECT too much for this request")],
    )

    result = controller.run_llama3_turn(stub, [{"role": "user", "content": "be careful"}])

    assert result == "I will avoid that and answer directly."
    assert stub.review_calls == 1


def test_read_file_is_capped(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_text("a" * 10_000)
    runner = ToolRunner(tmp_path / "memory.db")
    data = runner.read_file(str(target), max_chars=9999)
    assert len(data) == 4000


def test_grep_text_respects_match_limit(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("todo one\ntodo two\ntodo three\n")
    runner = ToolRunner(tmp_path / "memory.db")
    matches = runner.grep_text(str(target), pattern="todo", max_matches=2)
    assert len(matches) == 2


def test_file_management_tools_round_trip(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    source = tmp_path / "source.txt"
    source.write_text("hello tools")

    copy_result = runner.copy_file(str(source), str(tmp_path / "copied.txt"))
    assert "copied" in copy_result
    assert (tmp_path / "copied.txt").read_text() == "hello tools"

    move_result = runner.move_file(str(source), str(tmp_path / "moved.txt"))
    assert "moved" in move_result
    assert not source.exists()
    assert (tmp_path / "moved.txt").read_text() == "hello tools"

    rename_result = runner.rename_file(str(tmp_path / "moved.txt"), "renamed.txt")
    assert "renamed" in rename_result
    assert (tmp_path / "renamed.txt").read_text() == "hello tools"

    delete_result = runner.delete_file_safe(str(tmp_path / "renamed.txt"))
    assert "deleted file" in delete_result
    assert not (tmp_path / "renamed.txt").exists()


def test_organize_directory_supports_preview_and_apply(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path.parent / "memory.db")
    pdf = tmp_path / "report.pdf"
    txt = tmp_path / "notes.txt"
    pdf.write_text("pdf")
    txt.write_text("txt")

    preview = runner.organize_directory(str(tmp_path), dry_run=True)
    assert preview["planned_moves"] == 2
    assert pdf.exists()
    assert txt.exists()

    applied = runner.organize_directory(str(tmp_path), dry_run=False)
    assert applied["planned_moves"] == 2
    assert (tmp_path / "pdf" / "report.pdf").exists()
    assert (tmp_path / "txt" / "notes.txt").exists()


def test_organize_directory_defaults_to_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "photo.png").write_text("png")
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = ToolRunner(tmp_path / "memory.db")

    result = runner.organize_directory()

    assert result["path"] == str(downloads.resolve())
    assert result["planned_moves"] == 1


def test_memory_store_round_trip(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    runner.save_memory(key="goal", content="finish rexbot", namespace="user")
    results = runner.search_memory(query="rex", namespace="user")
    assert results[0]["key"] == "goal"


def test_knowledge_base_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("Rexbot knows Ollama and memory tools.")
    runner = ToolRunner(tmp_path / "memory.db")
    runner.add_knowledge(path=str(source), namespace="docs")
    results = runner.search_knowledge(query="Ollama", namespace="docs")
    assert results[0]["source_path"].endswith("notes.txt")


def test_skill_and_reflection_round_trip(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    runner.save_skill(name="refactor_python", steps="inspect, patch, test", outcome="clean commit")
    runner.log_reflection(task="refactor_python", reflection="bounded tools are easier to debug")
    assert runner.search_skills(query="refactor")[0]["key"] == "refactor_python"
    assert runner.list_reflections()[0]["key"] == "refactor_python"


def test_focus_and_persona_tools(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    runner.set_focus_mode(goal="ship rexbot")
    runner.set_personality_profile(name="anime_narrator", style="dramatic but concise")
    runner.set_emotion_state(state="focused", intensity="high")
    assert runner.get_focus_mode()[0]["key"] == "focus_mode"
    assert runner.get_personality_profile()[0]["key"] == "personality_profile"
    assert runner.get_emotion_state()[0]["key"] == "emotion_state"


def test_prototype_status_tools(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    assert runner.screen_vision_prototype()["status"] == "prototype_only"
    assert runner.pc_control_prototype()["status"] == "disabled_for_safety"


def test_manifest_and_status_tools(tmp_path: Path) -> None:
    runner = ToolRunner(tmp_path / "memory.db")
    manifest = runner.insane_tool_manifest()
    assert len(manifest) == 17
    assert runner.strategy_mode_status()["status"] == "working_with_llama3_planning"
    assert runner.autonomous_task_mode_status()["status"] == "working_partial"
    assert runner.multi_tool_chaining_status()["status"] == "working"
    assert runner.dual_brain_status()["status"] == "working_partial"



def test_dashboard_service_exposes_manifest_and_tool_runner(tmp_path: Path) -> None:
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    service = DashboardService(controller=controller)
    assert len(service.manifest()) == 17
    assert service.dashboard_state()["chat_enabled"] is False
    assert service.system_info()["python"]
    result = service.run_tool("write_file", {"path": str(tmp_path / "dash.txt"), "content": "ok"})
    assert "wrote" in result


def test_dashboard_chat_turn_returns_tool_events(tmp_path: Path) -> None:
    controller = AssistantController(
        policy=PolicyEngine(allowed_roots=[tmp_path], mode=SessionMode.SUPERVISED),
        audit_log=SessionAuditLog(tmp_path / "audit.json"),
        tool_runner=ToolRunner(tmp_path / "memory.db"),
    )
    target = tmp_path / "notes.txt"
    target.write_text("hello dashboard")
    stub = StubLlama3Client(
        responses=[
            Llama3Response(content="", tool_calls=[ToolCall("read_file", {"path": str(target)})]),
            Llama3Response(content="done", tool_calls=[]),
        ],
        reviews=[ReviewDecision(approved=True, content="CONFIRM on-task")],
    )
    service = DashboardService(controller=controller, client=stub)

    result = service.chat_turn("read the file")

    assert result["ok"] is True
    assert result["reply"] == "done"
    assert [event["event"] for event in result["tool_events"]] == [
        "tool_reviewed",
        "tool_requested",
        "tool_completed",
    ]


def test_registry_contains_expanded_tool_set(tmp_path: Path) -> None:
    registry = ToolRunner(tmp_path / "memory.db").build_registry()
    for tool_name in {
        "append_file",
        "make_dir",
        "move_file",
        "copy_file",
        "rename_file",
        "delete_file_safe",
        "organize_directory",
        "file_info",
        "directory_tree",
        "glob_search",
        "grep_text",
        "current_time",
        "system_info",
        "save_memory",
        "search_memory",
        "list_memories",
        "add_knowledge",
        "search_knowledge",
        "list_knowledge",
        "save_skill",
        "search_skills",
        "list_skills",
        "log_reflection",
        "list_reflections",
        "set_focus_mode",
        "get_focus_mode",
        "set_personality_profile",
        "get_personality_profile",
        "set_emotion_state",
        "get_emotion_state",
        "start_task_session",
        "get_task_session",
        "set_reviewer_mode",
        "get_reviewer_mode",
        "screen_vision_prototype",
        "object_detection_prototype",
        "pc_control_prototype",
        "code_execution_prototype",
        "game_controller_prototype",
        "voice_personality_prototype",
        "filesystem_intelligence_status",
        "strategy_mode_status",
        "autonomous_task_mode_status",
        "multi_tool_chaining_status",
        "self_improvement_status",
        "dual_brain_status",
        "insane_tool_manifest",
    }:
        assert tool_name in registry
