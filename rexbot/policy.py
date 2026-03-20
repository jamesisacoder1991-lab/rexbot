from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class SessionMode(str, Enum):
    SAFE = "safe"
    SUPERVISED = "supervised"


@dataclass(slots=True)
class Action:
    kind: str
    target: str
    detail: str = ""


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    needs_confirmation: bool
    reason: str


PATH_ACTIONS = {
    "list_dir",
    "read_file",
    "write_file",
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
    "add_knowledge",
}
NETWORK_ACTIONS = {"fetch_url"}
META_ACTIONS = {
    "current_time",
    "system_info",
    "save_memory",
    "search_memory",
    "list_memories",
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
}


@dataclass(slots=True)
class PolicyEngine:
    """Safety policy for a local assistant.

    SAFE mode requires confirmation for impactful file/network actions.
    SUPERVISED mode allows in-scope actions without repeated confirmation, but
    still blocks dangerous or out-of-scope actions.
    """

    allowed_roots: list[Path]
    allowed_domains: set[str] = field(default_factory=set)
    allow_network: bool = False
    mode: SessionMode = SessionMode.SAFE

    def __post_init__(self) -> None:
        self.allowed_roots = [root.expanduser().resolve() for root in self.allowed_roots]
        self.allowed_domains = {domain.lower() for domain in self.allowed_domains}

    def set_mode(self, mode: SessionMode) -> None:
        self.mode = mode

    def approve_roots(self, roots: Iterable[str | Path]) -> None:
        for root in roots:
            resolved = Path(root).expanduser().resolve()
            if resolved not in self.allowed_roots:
                self.allowed_roots.append(resolved)

    def approve_domains(self, domains: Iterable[str]) -> None:
        for domain in domains:
            self.allowed_domains.add(domain.lower())

    def evaluate(self, action: Action) -> PolicyDecision:
        if action.kind in PATH_ACTIONS:
            return self._evaluate_path_action(action)
        if action.kind in NETWORK_ACTIONS:
            return self._evaluate_network_action(action)
        if action.kind in META_ACTIONS:
            return PolicyDecision(True, False, "Lightweight metadata or memory tool is always allowed.")
        return PolicyDecision(False, False, f"Unknown action kind: {action.kind}")

    def _evaluate_path_action(self, action: Action) -> PolicyDecision:
        target_path = Path(action.target).expanduser().resolve()
        if not self._is_path_allowed(target_path):
            return PolicyDecision(
                False,
                False,
                f"Target path {target_path} is outside the approved roots.",
            )

        if self.mode is SessionMode.SAFE:
            return PolicyDecision(True, True, "Safe mode requires explicit confirmation.")

        return PolicyDecision(True, False, "Inside approved roots for supervised mode.")

    def _evaluate_network_action(self, action: Action) -> PolicyDecision:
        if not self.allow_network:
            return PolicyDecision(False, False, "Network access is disabled.")

        domain = action.target.lower().split("/", 1)[0]
        if self.allowed_domains and domain not in self.allowed_domains:
            return PolicyDecision(False, False, f"Domain {domain} is not approved.")

        if self.mode is SessionMode.SAFE:
            return PolicyDecision(True, True, "Safe mode requires explicit confirmation.")

        return PolicyDecision(True, False, "Approved domain for supervised mode.")

    def _is_path_allowed(self, path: Path) -> bool:
        for root in self.allowed_roots:
            if path == root or root in path.parents:
                return True
        return False
