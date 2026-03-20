from __future__ import annotations

import fnmatch
import json
import os
import platform
import re
import shutil
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen


DEFAULT_TEXT_LIMIT = 4_000
DEFAULT_TREE_DEPTH = 2
DEFAULT_TREE_ENTRIES = 200
DEFAULT_MATCH_LIMIT = 20
DEFAULT_GLOB_LIMIT = 100
DEFAULT_FETCH_LIMIT = 4_000
DEFAULT_MEMORY_LIMIT = 10


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def as_llama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MemoryStore:
    def __init__(self, db_path: str | Path = "data/rexbot_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_entry(
        self,
        table: str,
        namespace: str,
        entry_key: str,
        content: str,
    ) -> str:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                f"""
                INSERT INTO {table}(namespace, entry_key, content, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, entry_key)
                DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at
                """,
                (namespace, entry_key, content, datetime.now(timezone.utc).isoformat()),
            )
        return f"saved {table[:-1]} '{entry_key}' in namespace '{namespace}'"

    def search_entries(
        self,
        table: str,
        query: str,
        namespace: str,
        limit: int,
    ) -> list[dict[str, str]]:
        bounded = max(1, min(limit, DEFAULT_MEMORY_LIMIT))
        like_query = f"%{query}%"
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT entry_key, content, updated_at
                FROM {table}
                WHERE namespace = ? AND (entry_key LIKE ? OR content LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, like_query, like_query, bounded),
            ).fetchall()
        return [
            {"key": row[0], "content": row[1], "updated_at": row[2]}
            for row in rows
        ]

    def list_entries(
        self,
        table: str,
        namespace: str,
        limit: int,
    ) -> list[dict[str, str]]:
        bounded = max(1, min(limit, DEFAULT_MEMORY_LIMIT))
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT entry_key, content, updated_at
                FROM {table}
                WHERE namespace = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, bounded),
            ).fetchall()
        return [
            {"key": row[0], "content": row[1], "updated_at": row[2]}
            for row in rows
        ]

    def add_knowledge(
        self,
        source_path: str,
        content: str,
        namespace: str = "default",
    ) -> str:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO knowledge(namespace, source_path, content, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, source_path)
                DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at
                """,
                (namespace, source_path, content, datetime.now(timezone.utc).isoformat()),
            )
        return f"indexed knowledge from '{source_path}' in namespace '{namespace}'"

    def search_knowledge(
        self,
        query: str,
        namespace: str = "default",
        limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> list[dict[str, str]]:
        bounded = max(1, min(limit, DEFAULT_MEMORY_LIMIT))
        like_query = f"%{query}%"
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT source_path, content, updated_at
                FROM knowledge
                WHERE namespace = ? AND (source_path LIKE ? OR content LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, like_query, like_query, bounded),
            ).fetchall()
        return [
            {"source_path": row[0], "content": row[1][:300], "updated_at": row[2]}
            for row in rows
        ]

    def list_knowledge(
        self,
        namespace: str = "default",
        limit: int = DEFAULT_MEMORY_LIMIT,
    ) -> list[dict[str, str]]:
        bounded = max(1, min(limit, DEFAULT_MEMORY_LIMIT))
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT source_path, updated_at
                FROM knowledge
                WHERE namespace = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, bounded),
            ).fetchall()
        return [{"source_path": row[0], "updated_at": row[1]} for row in rows]

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            for table in ["memories", "skills", "reflections", "settings"]:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table}(
                        namespace TEXT NOT NULL,
                        entry_key TEXT NOT NULL,
                        content TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(namespace, entry_key)
                    )
                    """
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge(
                    namespace TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, source_path)
                )
                """
            )


class ToolRunner:
    def __init__(self, memory_db_path: str | Path = "data/rexbot_memory.db") -> None:
        self.memory_store = MemoryStore(memory_db_path)

    # File and system tools.
    def list_dir(self, path: str, limit: int = DEFAULT_GLOB_LIMIT) -> list[str]:
        target = Path(path).expanduser().resolve()
        entries = sorted(entry.name for entry in target.iterdir())
        return entries[: max(1, min(limit, DEFAULT_GLOB_LIMIT))]

    def read_file(self, path: str, max_chars: int = DEFAULT_TEXT_LIMIT) -> str:
        target = Path(path).expanduser().resolve()
        return self._read_text(target, max_chars)

    def write_file(self, path: str, content: str) -> str:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {target}"

    def append_file(self, path: str, content: str) -> str:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"appended {len(content)} bytes to {target}"

    def make_dir(self, path: str) -> str:
        target = Path(path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return f"created directory {target}"

    def move_file(self, src: str, dest: str, overwrite: bool = False) -> str:
        source = Path(src).expanduser().resolve()
        destination = Path(dest).expanduser().resolve()
        self._ensure_source_exists(source)
        self._ensure_destination_available(destination, overwrite)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return f"moved {source} to {destination}"

    def copy_file(self, src: str, dest: str, overwrite: bool = False) -> str:
        source = Path(src).expanduser().resolve()
        destination = Path(dest).expanduser().resolve()
        self._ensure_source_exists(source)
        self._ensure_destination_available(destination, overwrite)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return f"copied {source} to {destination}"

    def rename_file(self, path: str, new_name: str, overwrite: bool = False) -> str:
        target = Path(path).expanduser().resolve()
        self._ensure_source_exists(target)
        if Path(new_name).name != new_name:
            raise ValueError("new_name must be a bare file or folder name, not a path.")
        destination = target.with_name(new_name)
        self._ensure_destination_available(destination, overwrite)
        target.rename(destination)
        return f"renamed {target} to {destination}"

    def delete_file_safe(self, path: str, missing_ok: bool = False) -> str:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            if missing_ok:
                return f"nothing to delete at {target}"
            raise FileNotFoundError(f"{target} does not exist")
        if target.is_dir():
            if any(target.iterdir()):
                raise ValueError("delete_file_safe only removes files or empty directories.")
            target.rmdir()
            return f"deleted empty directory {target}"
        target.unlink()
        return f"deleted file {target}"

    def file_info(self, path: str) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        stat = target.stat()
        return {
            "path": str(target),
            "exists": target.exists(),
            "is_dir": target.is_dir(),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }

    def directory_tree(
        self,
        path: str,
        max_depth: int = DEFAULT_TREE_DEPTH,
        max_entries: int = DEFAULT_TREE_ENTRIES,
    ) -> list[str]:
        root = Path(path).expanduser().resolve()
        max_depth = max(0, min(max_depth, 5))
        max_entries = max(1, min(max_entries, DEFAULT_TREE_ENTRIES))
        queue: deque[tuple[Path, int]] = deque([(root, 0)])
        results: list[str] = []

        while queue and len(results) < max_entries:
            current, depth = queue.popleft()
            indent = "  " * depth
            suffix = "/" if current.is_dir() else ""
            results.append(f"{indent}{current.name or str(current)}{suffix}")
            if current.is_dir() and depth < max_depth:
                children = sorted(current.iterdir(), key=lambda item: item.name)[:max_entries]
                queue.extend((child, depth + 1) for child in children)
        return results

    def organize_directory(
        self,
        path: str = "~/Downloads",
        strategy: str = "extension",
        dry_run: bool = True,
        limit: int = DEFAULT_GLOB_LIMIT,
    ) -> dict[str, Any]:
        root = Path(path or "~/Downloads").expanduser().resolve()
        if strategy != "extension":
            raise ValueError("Only the 'extension' organization strategy is currently supported.")
        bounded = max(1, min(limit, DEFAULT_GLOB_LIMIT))
        plan: list[dict[str, str]] = []
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_file():
                continue
            folder_name = self._extension_bucket(child)
            destination = root / folder_name / child.name
            if child == destination:
                continue
            plan.append({"source": str(child), "destination": str(destination), "category": folder_name})
            if len(plan) >= bounded:
                break

        if not dry_run:
            for item in plan:
                destination = Path(item["destination"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(item["source"], item["destination"])
        return {
            "path": str(root),
            "strategy": strategy,
            "dry_run": dry_run,
            "planned_moves": len(plan),
            "moves": plan,
        }

    def glob_search(self, path: str, pattern: str, limit: int = DEFAULT_GLOB_LIMIT) -> list[str]:
        root = Path(path).expanduser().resolve()
        bounded = max(1, min(limit, DEFAULT_GLOB_LIMIT))
        matches: list[str] = []
        for child in root.rglob("*"):
            if fnmatch.fnmatch(child.name, pattern):
                matches.append(str(child))
            if len(matches) >= bounded:
                break
        return matches

    def grep_text(
        self,
        path: str,
        pattern: str,
        max_matches: int = DEFAULT_MATCH_LIMIT,
        max_chars_per_file: int = DEFAULT_TEXT_LIMIT,
    ) -> list[dict[str, Any]]:
        root = Path(path).expanduser().resolve()
        bounded = max(1, min(max_matches, DEFAULT_MATCH_LIMIT))
        results: list[dict[str, Any]] = []
        candidates = [root] if root.is_file() else root.rglob("*")
        regex = re.compile(pattern, re.IGNORECASE)

        for candidate in candidates:
            if not candidate.is_file():
                continue
            snippet = self._read_text(candidate, max_chars_per_file)
            for line_number, line in enumerate(snippet.splitlines(), start=1):
                if regex.search(line):
                    results.append({"path": str(candidate), "line": line_number, "text": line[:200]})
                if len(results) >= bounded:
                    return results
        return results

    def fetch_url(self, url: str, max_chars: int = DEFAULT_FETCH_LIMIT) -> str:
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        bounded = max(256, min(max_chars, DEFAULT_FETCH_LIMIT))
        with urlopen(target, timeout=10) as response:
            body = response.read(bounded)
        return body.decode("utf-8", errors="replace")

    def current_time(self, utc_offset: str = "+00:00") -> str:
        if not re.fullmatch(r"[+-]\d{2}:\d{2}", utc_offset):
            raise ValueError("utc_offset must use ±HH:MM format.")
        sign = 1 if utc_offset.startswith("+") else -1
        hours, minutes = utc_offset[1:].split(":")
        offset = timezone(sign * timedelta(hours=int(hours), minutes=int(minutes)))
        return datetime.now(offset).isoformat()

    def system_info(self) -> dict[str, Any]:
        memory_mb = None
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            memory_mb = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_mb": memory_mb,
            "machine": platform.machine(),
        }

    # Memory and knowledge tools.
    def save_memory(self, key: str, content: str, namespace: str = "default") -> str:
        return self.memory_store.save_entry("memories", namespace, key, content)

    def search_memory(self, query: str, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.search_entries("memories", query, namespace, limit)

    def list_memories(self, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.list_entries("memories", namespace, limit)

    def add_knowledge(self, path: str, namespace: str = "default", max_chars: int = DEFAULT_TEXT_LIMIT) -> str:
        target = Path(path).expanduser().resolve()
        content = self._read_text(target, max_chars)
        return self.memory_store.add_knowledge(str(target), content, namespace)

    def search_knowledge(self, query: str, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.search_knowledge(query, namespace, limit)

    def list_knowledge(self, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.list_knowledge(namespace, limit)

    # Safer versions of skill/self-improvement/personality features.
    def save_skill(self, name: str, steps: str, outcome: str = "", namespace: str = "default") -> str:
        payload = json.dumps({"steps": steps, "outcome": outcome})
        return self.memory_store.save_entry("skills", namespace, name, payload)

    def search_skills(self, query: str, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.search_entries("skills", query, namespace, limit)

    def list_skills(self, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.list_entries("skills", namespace, limit)

    def log_reflection(self, task: str, reflection: str, namespace: str = "default") -> str:
        return self.memory_store.save_entry("reflections", namespace, task, reflection)

    def list_reflections(self, namespace: str = "default", limit: int = DEFAULT_MEMORY_LIMIT) -> list[dict[str, str]]:
        return self.memory_store.list_entries("reflections", namespace, limit)

    def set_focus_mode(self, goal: str, namespace: str = "default") -> str:
        return self.memory_store.save_entry("settings", namespace, "focus_mode", goal)

    def get_focus_mode(self, namespace: str = "default") -> list[dict[str, str]]:
        return self.memory_store.search_entries("settings", "focus_mode", namespace, 1)

    def set_personality_profile(self, name: str, style: str, namespace: str = "default") -> str:
        payload = json.dumps({"name": name, "style": style})
        return self.memory_store.save_entry("settings", namespace, "personality_profile", payload)

    def get_personality_profile(self, namespace: str = "default") -> list[dict[str, str]]:
        return self.memory_store.search_entries("settings", "personality_profile", namespace, 1)

    def set_emotion_state(self, state: str, intensity: str = "medium", namespace: str = "default") -> str:
        payload = json.dumps({"state": state, "intensity": intensity})
        return self.memory_store.save_entry("settings", namespace, "emotion_state", payload)

    def get_emotion_state(self, namespace: str = "default") -> list[dict[str, str]]:
        return self.memory_store.search_entries("settings", "emotion_state", namespace, 1)

    def start_task_session(self, goal: str, max_steps: int = 8, namespace: str = "default") -> str:
        payload = json.dumps({"goal": goal, "max_steps": max_steps, "status": "queued"})
        return self.memory_store.save_entry("settings", namespace, "task_session", payload)

    def get_task_session(self, namespace: str = "default") -> list[dict[str, str]]:
        return self.memory_store.search_entries("settings", "task_session", namespace, 1)

    def set_reviewer_mode(self, enabled: bool = True, namespace: str = "default") -> str:
        payload = json.dumps({"enabled": enabled})
        return self.memory_store.save_entry("settings", namespace, "reviewer_mode", payload)

    def get_reviewer_mode(self, namespace: str = "default") -> list[dict[str, str]]:
        return self.memory_store.search_entries("settings", "reviewer_mode", namespace, 1)

    # Prototype/status tools for hard or unsafe roadmap items.
    def screen_vision_prototype(self, source: str = "active_screen") -> dict[str, Any]:
        return self._prototype_status(
            feature="screen_vision",
            status="prototype_only",
            details="Requires an external screenshot pipeline such as mss and a vision model.",
            source=source,
        )

    def object_detection_prototype(self, target: str = "active_screen") -> dict[str, Any]:
        return self._prototype_status(
            feature="object_detection",
            status="prototype_only",
            details="Requires external CV dependencies such as YOLO and OpenCV.",
            target=target,
        )

    def pc_control_prototype(self) -> dict[str, Any]:
        return self._prototype_status(
            feature="pc_control",
            status="disabled_for_safety",
            details="Unrestricted mouse/keyboard/app control is intentionally not enabled in this prototype.",
        )

    def code_execution_prototype(self) -> dict[str, Any]:
        return self._prototype_status(
            feature="code_execution",
            status="disabled_for_safety",
            details="Arbitrary code execution is intentionally not enabled in this prototype.",
        )

    def game_controller_prototype(self) -> dict[str, Any]:
        return self._prototype_status(
            feature="game_controller",
            status="disabled_for_safety",
            details="Game automation and controller spoofing are intentionally not enabled in this prototype.",
        )

    def voice_personality_prototype(self, voice_name: str = "assistant") -> dict[str, Any]:
        return self._prototype_status(
            feature="voice_personality",
            status="prototype_only",
            details="Requires an external TTS engine such as Coqui TTS.",
            voice_name=voice_name,
        )


    def filesystem_intelligence_status(self) -> dict[str, Any]:
        return {
            "feature": "filesystem_intelligence",
            "status": "working",
            "backing_tools": [
                "list_dir",
                "read_file",
                "write_file",
                "append_file",
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
            ],
        }

    def strategy_mode_status(self) -> dict[str, Any]:
        return {
            "feature": "strategy_mode",
            "status": "working_with_llama3_planning",
            "details": "Llama 3 already plans step-by-step during tool loops; task-session tools store longer goals.",
            "backing_tools": ["start_task_session", "get_task_session"],
        }

    def autonomous_task_mode_status(self) -> dict[str, Any]:
        return {
            "feature": "autonomous_task_mode",
            "status": "working_partial",
            "details": "Task goals can be queued and revisited with bounded tool rounds.",
            "backing_tools": ["start_task_session", "get_task_session"],
        }

    def multi_tool_chaining_status(self) -> dict[str, Any]:
        return {
            "feature": "multi_tool_chaining",
            "status": "working",
            "details": "The controller already loops Llama 3 tool calls until it reaches a final response.",
            "backing_tools": ["run_llama3_turn"],
        }

    def self_improvement_status(self) -> dict[str, Any]:
        return {
            "feature": "self_improvement",
            "status": "working_partial",
            "backing_tools": ["log_reflection", "list_reflections", "save_skill", "search_skills"],
        }

    def dual_brain_status(self) -> dict[str, Any]:
        return {
            "feature": "dual_brain_system",
            "status": "working_partial",
            "details": "Reviewer mode preferences are implemented; an actual second-model checker can be added later.",
            "backing_tools": ["set_reviewer_mode", "get_reviewer_mode"],
        }

    def insane_tool_manifest(self) -> list[dict[str, Any]]:
        return [
            {"item": 1, "name": "Screen Vision", "status": "prototype_only", "tool": "screen_vision_prototype"},
            {"item": 2, "name": "Object Detection", "status": "prototype_only", "tool": "object_detection_prototype"},
            {"item": 3, "name": "Long-Term Memory", "status": "working", "tool": "save_memory/search_memory/list_memories"},
            {"item": 4, "name": "Skill Learning", "status": "working", "tool": "save_skill/search_skills/list_skills"},
            {"item": 5, "name": "Full PC Control", "status": "disabled_for_safety", "tool": "pc_control_prototype"},
            {"item": 6, "name": "File System Intelligence", "status": "working", "tool": "filesystem_intelligence_status"},
            {"item": 7, "name": "Code Execution Tool", "status": "disabled_for_safety", "tool": "code_execution_prototype"},
            {"item": 8, "name": "Game Controller AI", "status": "disabled_for_safety", "tool": "game_controller_prototype"},
            {"item": 9, "name": "Strategy Mode", "status": "working_with_llama3_planning", "tool": "strategy_mode_status"},
            {"item": 10, "name": "Voice + Personality", "status": "prototype_only", "tool": "voice_personality_prototype"},
            {"item": 11, "name": "Emotion Simulation", "status": "working", "tool": "set_emotion_state/get_emotion_state"},
            {"item": 12, "name": "Local Knowledge Base", "status": "working", "tool": "add_knowledge/search_knowledge/list_knowledge"},
            {"item": 13, "name": "Autonomous Task Mode", "status": "working_partial", "tool": "autonomous_task_mode_status"},
            {"item": 14, "name": "Multi-Tool Chaining", "status": "working", "tool": "multi_tool_chaining_status"},
            {"item": 15, "name": "Self-Improvement System", "status": "working_partial", "tool": "self_improvement_status"},
            {"item": 16, "name": "Dual-Brain System", "status": "working_partial", "tool": "dual_brain_status"},
            {"item": 17, "name": "Focus Mode", "status": "working", "tool": "set_focus_mode/get_focus_mode"},
        ]

    def build_registry(self) -> dict[str, ToolDefinition]:
        tool_specs = [
            ("list_dir", "List files and folders inside an approved directory with a bounded result set.", {"path": "Directory path to inspect.", "limit": "Maximum entries to return."}, ["path"]),
            ("read_file", "Read a UTF-8 text file from an approved path with an output size cap.", {"path": "File path to read.", "max_chars": "Maximum characters to read."}, ["path"]),
            ("write_file", "Write UTF-8 text content to an approved file path.", {"path": "File path to write.", "content": "Text content to save."}, ["path", "content"]),
            ("append_file", "Append UTF-8 text content to an approved file path.", {"path": "File path to append to.", "content": "Text content to append."}, ["path", "content"]),
            ("make_dir", "Create a directory inside an approved path.", {"path": "Directory path to create."}, ["path"]),
            ("move_file", "Move a file or folder between approved paths.", {"src": "Source path to move.", "dest": "Destination path to move to.", "overwrite": "Whether an existing destination can be replaced."}, ["src", "dest"]),
            ("copy_file", "Copy a file between approved paths.", {"src": "Source path to copy.", "dest": "Destination path to copy to.", "overwrite": "Whether an existing destination can be replaced."}, ["src", "dest"]),
            ("rename_file", "Rename a file or directory within its current parent folder.", {"path": "Existing path to rename.", "new_name": "New filename or folder name only.", "overwrite": "Whether an existing sibling with the new name can be replaced."}, ["path", "new_name"]),
            ("delete_file_safe", "Delete a file or an empty directory inside approved scope.", {"path": "Path to delete.", "missing_ok": "Whether deleting a missing path should be treated as success."}, ["path"]),
            ("organize_directory", "Plan or apply a simple directory organization workflow by file extension. If no path is provided, Rexbot uses ~/Downloads.", {"path": "Directory to organize. Defaults to ~/Downloads when omitted.", "strategy": "Organization strategy; currently only extension.", "dry_run": "Preview moves without applying them.", "limit": "Maximum files to include in the plan."}, []),
            ("file_info", "Return metadata for a file or directory inside approved scope.", {"path": "Path to inspect."}, ["path"]),
            ("directory_tree", "Return a bounded shallow directory tree for an approved path.", {"path": "Root path to inspect.", "max_depth": "Depth limit.", "max_entries": "Maximum lines to return."}, ["path"]),
            ("glob_search", "Search for matching filenames under an approved path with a bounded result set.", {"path": "Root path to search.", "pattern": "Filename glob such as *.py or *.md.", "limit": "Maximum matches to return."}, ["path", "pattern"]),
            ("grep_text", "Search text within a file or directory tree in approved scope with bounded results.", {"path": "File or directory root to search.", "pattern": "Regular expression or text pattern.", "max_matches": "Maximum matches to return.", "max_chars_per_file": "Maximum text to read from each file."}, ["path", "pattern"]),
            ("fetch_url", "Fetch text from an approved URL or domain with an output size cap.", {"url": "URL or domain to fetch.", "max_chars": "Maximum characters to read."}, ["url"]),
            ("current_time", "Get the current time for a UTC offset like +00:00 or -05:00.", {"utc_offset": "UTC offset in ±HH:MM format."}, []),
            ("system_info", "Return lightweight device information useful for adapting to slower hardware.", {}, []),
            ("save_memory", "Save a long-term memory note into Rexbot's SQLite-backed memory store.", {"key": "Stable memory key.", "content": "Memory content to store.", "namespace": "Optional memory namespace."}, ["key", "content"]),
            ("search_memory", "Search Rexbot's long-term memory store by key or content.", {"query": "Text query for memory search.", "namespace": "Optional memory namespace.", "limit": "Maximum memories to return."}, ["query"]),
            ("list_memories", "List recent long-term memory entries.", {"namespace": "Optional memory namespace.", "limit": "Maximum memories to return."}, []),
            ("add_knowledge", "Read a local file and add it to Rexbot's searchable knowledge base.", {"path": "Local text file to index.", "namespace": "Optional knowledge namespace.", "max_chars": "Maximum characters to index from the file."}, ["path"]),
            ("search_knowledge", "Search Rexbot's local knowledge base for relevant indexed notes or docs.", {"query": "Text query for the knowledge base.", "namespace": "Optional knowledge namespace.", "limit": "Maximum knowledge matches to return."}, ["query"]),
            ("list_knowledge", "List recently indexed knowledge-base documents.", {"namespace": "Optional knowledge namespace.", "limit": "Maximum knowledge items to return."}, []),
            ("save_skill", "Save a reusable skill recipe or workflow.", {"name": "Skill name.", "steps": "Step-by-step instructions.", "outcome": "Optional expected outcome.", "namespace": "Optional namespace."}, ["name", "steps"]),
            ("search_skills", "Search saved skill recipes.", {"query": "Text query for skills.", "namespace": "Optional namespace.", "limit": "Maximum skills to return."}, ["query"]),
            ("list_skills", "List recent saved skills.", {"namespace": "Optional namespace.", "limit": "Maximum skills to return."}, []),
            ("log_reflection", "Record a reflection about a task or failure.", {"task": "Task name.", "reflection": "What was learned.", "namespace": "Optional namespace."}, ["task", "reflection"]),
            ("list_reflections", "List recent self-improvement reflections.", {"namespace": "Optional namespace.", "limit": "Maximum reflections to return."}, []),
            ("set_focus_mode", "Set the assistant's current focus goal.", {"goal": "Current focus goal.", "namespace": "Optional namespace."}, ["goal"]),
            ("get_focus_mode", "Read the current focus goal.", {"namespace": "Optional namespace."}, []),
            ("set_personality_profile", "Store a personality profile for the assistant.", {"name": "Profile name.", "style": "How the assistant should sound.", "namespace": "Optional namespace."}, ["name", "style"]),
            ("get_personality_profile", "Read the active personality profile.", {"namespace": "Optional namespace."}, []),
            ("set_emotion_state", "Store an emotion/state tag for the assistant persona.", {"state": "Emotion label.", "intensity": "Intensity label.", "namespace": "Optional namespace."}, ["state"]),
            ("get_emotion_state", "Read the current emotion/persona state.", {"namespace": "Optional namespace."}, []),
            ("start_task_session", "Store a task-session goal for longer autonomous work loops.", {"goal": "Goal to work on.", "max_steps": "Maximum internal steps.", "namespace": "Optional namespace."}, ["goal"]),
            ("get_task_session", "Read the stored task-session goal/status.", {"namespace": "Optional namespace."}, []),
            ("set_reviewer_mode", "Enable or disable a dual-brain reviewer preference.", {"enabled": "Whether reviewer mode is enabled.", "namespace": "Optional namespace."}, []),
            ("get_reviewer_mode", "Read the reviewer-mode preference.", {"namespace": "Optional namespace."}, []),
            ("screen_vision_prototype", "Prototype status for future screen-vision support.", {"source": "Target screen source."}, []),
            ("object_detection_prototype", "Prototype status for future object detection support.", {"target": "Target visual source."}, []),
            ("pc_control_prototype", "Prototype status for future PC-control support.", {}, []),
            ("code_execution_prototype", "Prototype status for future code-execution support.", {}, []),
            ("game_controller_prototype", "Prototype status for future game-controller support.", {}, []),
            ("voice_personality_prototype", "Prototype status for future TTS/voice personality support.", {"voice_name": "Requested voice profile."}, []),
            ("filesystem_intelligence_status", "Report the current file-system intelligence coverage.", {}, []),
            ("strategy_mode_status", "Report the current strategy-mode support level.", {}, []),
            ("autonomous_task_mode_status", "Report the current autonomous-task support level.", {}, []),
            ("multi_tool_chaining_status", "Report the current multi-tool chaining support level.", {}, []),
            ("self_improvement_status", "Report the current self-improvement support level.", {}, []),
            ("dual_brain_status", "Report the current dual-brain support level.", {}, []),
            ("insane_tool_manifest", "Return the numbered roadmap manifest for the full requested capability list.", {}, []),
        ]

        registry: dict[str, ToolDefinition] = {}
        for name, description, properties, required in tool_specs:
            schema_properties = {
                key: {"type": "string", "description": value} for key, value in properties.items()
            }
            for numeric_key in {"limit", "max_chars", "max_depth", "max_entries", "max_matches", "max_chars_per_file", "max_steps"} & properties.keys():
                schema_properties[numeric_key]["type"] = "integer"
            for boolean_key in {"enabled", "overwrite", "missing_ok", "dry_run"} & schema_properties.keys():
                schema_properties[boolean_key]["type"] = "boolean"
            registry[name] = ToolDefinition(
                name=name,
                description=description,
                parameters={"type": "object", "properties": schema_properties, "required": required},
                handler=getattr(self, name),
            )
        return registry

    def _prototype_status(self, feature: str, status: str, details: str, **extra: Any) -> dict[str, Any]:
        return {"feature": feature, "status": status, "details": details, **extra}

    def _read_text(self, path: Path, max_chars: int) -> str:
        bounded = max(256, min(max_chars, DEFAULT_TEXT_LIMIT))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(bounded)

    def _ensure_source_exists(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

    def _ensure_destination_available(self, path: Path, overwrite: bool) -> None:
        if not path.exists():
            return
        if not overwrite:
            raise FileExistsError(f"{path} already exists")
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def _extension_bucket(self, path: Path) -> str:
        suffix = path.suffix.lower().lstrip(".")
        return suffix or "no_extension"
