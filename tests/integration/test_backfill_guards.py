"""--backfill refuses what could delete or double-count the wrong traces."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from conftest import FakeLangfuse


SESSION = "11111111-2222-3333-4444-555555555555"


def write_transcript(directory: Path, session_id: str = SESSION, *, age_minutes: float = 120) -> Path:
    path = directory / f"{session_id}.jsonl"
    rows = [
        {"type": "user", "uuid": "u1", "sessionId": session_id, "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"role": "user", "content": "hi"}},
        {"type": "assistant", "uuid": "a1", "sessionId": session_id, "timestamp": "2026-08-01T10:00:01.000Z",
         "message": {"id": "m1", "role": "assistant", "content": [{"type": "text", "text": "hello"}],
                     "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 2}}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    then = time.time() - age_minutes * 60
    os.utime(path, (then, then))
    return path


class FakeTraceApi:
    def __init__(self, journal: List[str], traces: List[Any]) -> None:
        self.journal = journal
        self.traces = traces
        self.deleted: List[str] = []

    def list(self, **kwargs: Any) -> Any:
        self.journal.append("list")
        return SimpleNamespace(data=list(self.traces), meta=SimpleNamespace(total_pages=1))

    def delete_multiple(self, *, trace_ids: List[str]) -> None:
        self.journal.append("delete")
        self.deleted += list(trace_ids)


class RecordingClient(FakeLangfuse):
    def __init__(self, traces: List[Any]) -> None:
        super().__init__()
        self.journal: List[str] = []
        self.api = SimpleNamespace(trace=FakeTraceApi(self.journal, traces))

    def flush(self) -> None:
        self.journal.append(f"flush:{len(self.observations)}")

    def shutdown(self) -> None:
        pass


def trace(trace_id: str, name: Any) -> Any:
    return SimpleNamespace(id=trace_id, name=name)


@pytest.fixture
def client(hook_module, monkeypatch):
    def make(traces: List[Any]) -> RecordingClient:
        instance = RecordingClient(traces)
        config = hook_module.LangfuseConfig("pk", "sk", "https://example.test", None)
        monkeypatch.setattr(hook_module, "get_langfuse_config", lambda: config)
        monkeypatch.setattr(hook_module, "create_langfuse_client", lambda c: instance)
        monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", None)
        return instance
    return make


def test_mismatched_session_id_is_refused_and_nothing_is_touched(hook_module, client, tmp_path):
    transcript = write_transcript(tmp_path)
    c = client([trace("o1", hook_module.TRACE_NAME)])
    code = hook_module.run_backfill(["--backfill", str(transcript), "--session-id", "OTHER",
                                     "--replace-session-traces"])
    assert code == 3
    assert c.api.trace.deleted == [] and c.journal == []


def test_transcript_named_for_another_session_is_refused(hook_module, client, tmp_path):
    # Rows belong to SESSION, but the file (so the live traces) is another's.
    transcript = write_transcript(tmp_path)
    renamed = tmp_path / "99999999-0000-0000-0000-000000000000.jsonl"
    renamed.write_text(transcript.read_text(encoding="utf-8"), encoding="utf-8")
    then = time.time() - 7200
    os.utime(renamed, (then, then))
    c = client([trace("o1", hook_module.TRACE_NAME)])
    assert hook_module.run_backfill(["--backfill", str(renamed), "--session-id", SESSION,
                                     "--replace-session-traces"]) == 3
    assert c.api.trace.deleted == [] and c.observations == []


def test_recently_modified_transcript_is_refused_without_force(hook_module, client, tmp_path):
    transcript = write_transcript(tmp_path, age_minutes=1)
    c = client([])
    assert hook_module.run_backfill(["--backfill", str(transcript)]) == 3
    assert c.observations == []
    assert hook_module.run_backfill(["--backfill", str(transcript), "--force"]) == 0
    assert c.observations


def test_upload_happens_before_delete_and_only_hook_traces_go(hook_module, client, tmp_path):
    transcript = write_transcript(tmp_path)
    c = client([trace("old-turn", hook_module.TRACE_NAME), trace("foreign", "Some other app")])
    assert hook_module.run_backfill(["--backfill", str(transcript), "--replace-session-traces"]) == 0
    assert c.api.trace.deleted == ["old-turn"]
    first_delete = c.journal.index("delete")
    flushes_before = [j for j in c.journal[:first_delete] if j.startswith("flush:")]
    assert flushes_before and int(flushes_before[-1].split(":")[1]) > 0, c.journal


def test_the_backfills_own_traces_are_kept(hook_module, client, tmp_path):
    transcript = write_transcript(tmp_path)
    c = client([])
    keep = hook_module.get_backfill_trace_ids(c, SESSION, transcript)
    # keep is computed under the default namespace the run will set.
    hook_module.TRACE_SEED_NAMESPACE = hook_module.DEFAULT_BACKFILL_NAMESPACE
    keep = hook_module.get_backfill_trace_ids(c, SESSION, transcript)
    plan = hook_module.plan_trace_replacement(
        [(t, hook_module.TRACE_NAME) for t in keep] + [("old", hook_module.TRACE_NAME)], keep)
    assert plan["delete"] == ["old"] and sorted(plan["keep"]) == sorted(keep)


def test_second_run_in_the_same_namespace_is_refused(hook_module, client, tmp_path):
    transcript = write_transcript(tmp_path)
    c = client([])
    assert hook_module.run_backfill(["--backfill", str(transcript)]) == 0
    uploaded = {o.trace_id for o in c.observations}
    c.api.trace.traces = [trace(t, hook_module.TRACE_NAME) for t in uploaded]
    before = len(c.observations)
    assert hook_module.run_backfill(["--backfill", str(transcript)]) == 3
    assert len(c.observations) == before


def test_dry_run_lists_the_delete_plan_and_deletes_nothing(hook_module, client, tmp_path, capsys):
    transcript = write_transcript(tmp_path)
    c = client([trace("old-turn", hook_module.TRACE_NAME), trace("foreign", None)])
    assert hook_module.run_backfill(["--backfill", str(transcript), "--dry-run",
                                     "--replace-session-traces"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["would_delete"] == ["old-turn"]
    assert out["replace_plan"] == {"delete": 1, "keep": 0, "foreign": 1}
    assert c.api.trace.deleted == [] and c.observations == []
