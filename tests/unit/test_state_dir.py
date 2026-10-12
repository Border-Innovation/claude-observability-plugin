from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any

import pytest


OPTION = "CC_LANGFUSE_STATE_DIR"


@pytest.fixture(autouse=True)
def clean_state_dir_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPTION, raising=False)
    monkeypatch.delenv(f"CLAUDE_PLUGIN_OPTION_{OPTION}", raising=False)


def _assert_actionable_fallback(warning: str, rejected_value: str) -> None:
    # A fallback warning is only useful if it names the option, the rejected
    # value and where state actually went instead.
    assert OPTION in warning
    assert rejected_value in warning
    assert str(Path.home() / ".claude" / "state") in warning


# ----------------- accepted values -----------------

def test_default_without_override(hook_module: Any) -> None:
    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == Path.home() / ".claude" / "state"
    assert warning == ""


def test_empty_value_counts_as_unset(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPTION, "")

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == Path.home() / ".claude" / "state"
    assert warning == ""


def test_plain_env_var_override(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "profile-a" / "state"
    monkeypatch.setenv(OPTION, str(target))

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == target
    assert warning == ""
    assert target.is_dir()


def test_plugin_option_channel(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "wizard-state"
    monkeypatch.setenv(f"CLAUDE_PLUGIN_OPTION_{OPTION}", str(target))

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == target
    assert warning == ""
    assert target.is_dir()


def test_tilde_expands_against_home(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(OPTION, "~/agent/state")

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == tmp_path / "agent" / "state"
    assert warning == ""


# ----------------- rejected values fall back loudly -----------------

@pytest.mark.parametrize("value", ["relative/state", "./state", " "])
def test_non_absolute_values_fall_back(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(OPTION, value)

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == Path.home() / ".claude" / "state"
    assert "not an absolute path" in warning
    _assert_actionable_fallback(warning, value)
    # The rejected value must not leave a directory behind in the hook's cwd.
    assert not (Path.cwd() / "relative").exists()


def test_unknown_user_tilde_falls_back(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Path.expanduser raises RuntimeError for an unresolvable ~user prefix;
    # that must degrade to the default, not crash the module import.
    monkeypatch.setenv(OPTION, "~no-such-user-xyz/state")

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == Path.home() / ".claude" / "state"
    assert "unusable" in warning
    _assert_actionable_fallback(warning, "~no-such-user-xyz/state")


@pytest.mark.skipif(sys.platform == "win32", reason="permission bits are POSIX semantics")
def test_existing_unwritable_dir_falls_back(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # mkdir(exist_ok=True) succeeds on an existing read-only dir, so the probe
    # alone would accept a directory in which log, lock and state all fail.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses permission checks")
    ro = tmp_path / "ro-state"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        monkeypatch.setenv(OPTION, str(ro))

        state_dir, warning = hook_module._resolve_state_dir()

        assert state_dir == Path.home() / ".claude" / "state"
        assert "unusable" in warning
        assert "not writable" in warning
        _assert_actionable_fallback(warning, str(ro))
    finally:
        ro.chmod(0o755)


def test_uncreatable_path_falls_back(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    target = blocker / "state"
    monkeypatch.setenv(OPTION, str(target))

    state_dir, warning = hook_module._resolve_state_dir()

    assert state_dir == Path.home() / ".claude" / "state"
    assert "unusable" in warning
    _assert_actionable_fallback(warning, str(target))


# ----------------- import-time wiring -----------------

def _fresh_hook_module(name: str) -> Any:
    """Import a private module instance so the import-time path resolution
    (STATE_DIR, _STATE_DIR_WARNING = _resolve_state_dir()) runs under the
    current test environment. The shared hook_module fixture imports once per
    session, so it can never observe env vars set inside a test."""
    module_path = Path(__file__).resolve().parents[2] / "hooks" / "langfuse_hook.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_import_wires_globals_to_override(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # hook_module guarantees the langfuse stubs are installed for the fresh import.
    target = tmp_path / "profile" / "state"
    monkeypatch.setenv(OPTION, str(target))

    fresh = _fresh_hook_module("langfuse_hook_wiring_override")

    assert fresh.STATE_DIR == target
    assert fresh._STATE_DIR_WARNING == ""
    assert fresh.LOG_FILE == target / "langfuse_hook.log"
    assert fresh.STATE_FILE == target / "langfuse_state.json"
    assert fresh.LOCK_FILE == target / "langfuse_state.lock"


def test_import_wires_fallback_and_warning(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))  # keep the fallback away from the real home
    monkeypatch.setenv(OPTION, "relative/state")

    fresh = _fresh_hook_module("langfuse_hook_wiring_fallback")

    assert fresh.STATE_DIR == tmp_path / ".claude" / "state"
    assert "not an absolute path" in fresh._STATE_DIR_WARNING


# ----------------- the warning reaches the log -----------------

def test_main_logs_warning_even_without_langfuse_keys(
    hook_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook_module, "_STATE_DIR_WARNING", "CC_LANGFUSE_STATE_DIR test-warning")
    for name in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "CC_LANGFUSE_PUBLIC_KEY",
        "CC_LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"CLAUDE_PLUGIN_OPTION_{name}", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert hook_module.main() == 0

    log = Path(hook_module.LOG_FILE).read_text(encoding="utf-8")
    assert "CC_LANGFUSE_STATE_DIR test-warning" in log
    assert "[INFO]" in log
