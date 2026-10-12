from __future__ import annotations

from typing import Any

import pytest


class _Turn:
    def __init__(self, uuid: str) -> None:
        self.user_msg = {"uuid": uuid}


class _FlushCountingLangfuse:
    def __init__(self) -> None:
        self.flushes = 0

    def flush(self) -> None:
        self.flushes += 1


def _emit(hook_module: Any, monkeypatch: pytest.MonkeyPatch, n_turns: int) -> _FlushCountingLangfuse:
    monkeypatch.setattr(hook_module, "emit_turn", lambda *a, **k: {})
    langfuse = _FlushCountingLangfuse()
    state = hook_module.SessionState()
    turns = [_Turn(f"u{i}") for i in range(n_turns)]
    state.turn_numbers = {f"u{i}": i + 1 for i in range(n_turns)}
    emitted = hook_module.emit_and_close_ready_turns(
        langfuse, "s", None, turns, state,
        user_id=None, subagent_transcripts_by_tool_use_id={},
    )
    assert emitted == n_turns
    return langfuse


def test_backlog_firing_flushes_after_every_turn(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _emit(hook_module, monkeypatch, 3).flushes == 3


def test_single_turn_firing_does_not_flush_in_loop(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _emit(hook_module, monkeypatch, 1).flushes == 0


def test_client_creation_raises_otel_queue_ceiling(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_BSP_MAX_QUEUE_SIZE", raising=False)
    seen: dict[str, Any] = {}

    class RecordingLangfuse:
        def __init__(self, **_: Any) -> None:
            import os
            seen["queue"] = os.environ.get("OTEL_BSP_MAX_QUEUE_SIZE")

    monkeypatch.setattr(hook_module, "Langfuse", RecordingLangfuse)
    config = hook_module.LangfuseConfig(public_key="pk", secret_key="sk", host="h", user_id=None)
    assert hook_module.create_langfuse_client(config) is not None
    assert seen["queue"] == hook_module.DEFAULT_OTEL_BSP_MAX_QUEUE_SIZE


def test_client_creation_keeps_user_queue_ceiling(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "123")
    monkeypatch.setattr(hook_module, "Langfuse", lambda **_: object())
    config = hook_module.LangfuseConfig(public_key="pk", secret_key="sk", host="h", user_id=None)
    hook_module.create_langfuse_client(config)
    import os
    assert os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] == "123"


def test_client_creation_caps_spans_per_export_request(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_FLUSH_AT", raising=False)
    seen: dict[str, Any] = {}

    class RecordingLangfuse:
        def __init__(self, **_: Any) -> None:
            import os
            seen["flush_at"] = os.environ.get("LANGFUSE_FLUSH_AT")

    monkeypatch.setattr(hook_module, "Langfuse", RecordingLangfuse)
    config = hook_module.LangfuseConfig(public_key="pk", secret_key="sk", host="h", user_id=None)
    hook_module.create_langfuse_client(config)
    assert seen["flush_at"] == hook_module.DEFAULT_LANGFUSE_FLUSH_AT
