"""Nested and resumed agents reach Langfuse whole, once, with their usage.

Fixture (synthetic rows in the Claude Code 2.1.301 shapes):

    main --bg--> orch --bg--> child1 --sync--> grand        (depth 3)
                     \\--Workflow wf_t1 --> w1 --sync--> wchild
                     \\--bg--> child2                       (phase 2 only)

phase1 is the disk at the first Stop: orch has ended a turn while child1
still runs, so main got an interim "completed" notification for orch. phase2
is appended before the second Stop: orch resumed on child1's notification,
launched child2 and finished. Every transcript also carries api-request /
api-request-blob echo rows.

The "server view" dedupes observations by id with the last copy winning,
which is how Langfuse stores an observation exported twice.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional



FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "resumed_background_agent"
SESSION = "session-resume"
AGENTS = ["orch", "child1", "grand", "child2", "w1", "wchild"]


def apply_phase(phase: str, dst: Path) -> None:
    """Append every file of the phase to the same relative path under dst."""
    src_root = FIXTURE / phase
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        target = dst / src.relative_to(src_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.name.endswith(".meta.json"):
            shutil.copyfile(src, target)
            continue
        with target.open("a", encoding="utf-8") as f:
            f.write(src.read_text(encoding="utf-8"))


def fire(hook_module: Any, fake_langfuse: Any, transcript: Path, *, end: bool = False) -> None:
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, SESSION, transcript, flush_deferred_agent_turns=end,
    )


def run_session(hook_module: Any, fake_langfuse: Any, tmp_path: Path) -> Path:
    apply_phase("phase1", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    fire(hook_module, fake_langfuse, transcript)
    apply_phase("phase2", tmp_path)
    fire(hook_module, fake_langfuse, transcript)
    fire(hook_module, fake_langfuse, transcript, end=True)
    return transcript


def server_view(fake_langfuse: Any) -> Dict[str, Any]:
    view: Dict[str, Any] = {}
    for observation in fake_langfuse.observations:
        view[observation.id] = observation
    return view


def parent_id(observation: Any) -> Optional[str]:
    parent = observation._otel_span.parent
    if parent is None:
        return None
    return f"{parent.get_span_context().span_id:016x}"


def agent_of(observation: Any, view: Dict[str, Any]) -> str:
    current: Optional[Any] = observation
    while current is not None:
        if current.as_type == "span" and current.name.startswith(("Subagent", "Workflow agent")):
            path = current.kwargs.get("metadata", {}).get("transcript_path") or ""
            return path[len("agent-"):-len(".jsonl")]
        current = view.get(parent_id(current) or "")
    return "main"


def agent_spans(view: Dict[str, Any]) -> Dict[str, List[str]]:
    spans: Dict[str, List[str]] = defaultdict(list)
    for observation in view.values():
        if observation.as_type == "span" and observation.name.startswith(("Subagent", "Workflow agent")):
            path = observation.kwargs.get("metadata", {}).get("transcript_path") or ""
            spans[path[len("agent-"):-len(".jsonl")]].append(observation.id)
    return spans


def transcript_usage(tmp_path: Path, agent: str) -> Dict[str, Any]:
    """Independent oracle: the agent's own transcript, echo rows skipped,
    one entry per message.id, the LAST row's usage (the complete one)."""
    path = next(tmp_path.rglob(f"agent-{agent}.jsonl"))
    by_message: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") != "assistant":
            continue
        by_message[row["message"]["id"]] = row["message"]["usage"]
    totals: Counter = Counter()
    for u in by_message.values():
        totals["output"] += u["output_tokens"]
        totals["cache_read_input_tokens"] += u["cache_read_input_tokens"]
    return {"messages": len(by_message), "totals": totals}


def emitted_usage(view: Dict[str, Any], agent: str) -> Dict[str, Any]:
    generations = [o for o in view.values() if o.as_type == "generation" and agent_of(o, view) == agent]
    totals: Counter = Counter()
    zero = 0
    for g in generations:
        usage = g.kwargs.get("usage_details") or {}
        if not usage:
            zero += 1
        totals["output"] += usage.get("output", 0)
        totals["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
    return {"messages": len(generations), "totals": totals, "zero": zero}


# ---------------------------------------------------------------- the fix

def test_every_agent_at_every_depth_arrives_exactly_once(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(server_view(fake_langfuse))
    for agent in AGENTS:
        assert len(spans.get(agent, [])) == 1, f"{agent}: {spans.get(agent)}"


def test_each_agent_carries_every_message_once_with_its_final_usage(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    view = server_view(fake_langfuse)
    for agent in AGENTS:
        expected = transcript_usage(tmp_path, agent)
        got = emitted_usage(view, agent)
        assert got["zero"] == 0, f"{agent}: zero-usage generations"
        assert got["messages"] == expected["messages"], agent
        assert got["totals"] == expected["totals"], agent


def test_resumed_agent_span_is_replaced_not_duplicated(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    exported = [o for o in fake_langfuse.observations
                if o.as_type == "span" and o.name == "Subagent: Batch orchestrator"]
    # Exported at the first Stop (interim) and again once it grew ...
    assert len(exported) >= 2
    # ... always under one id, the last copy reaching the agent's real end.
    assert len({o.id for o in exported}) == 1
    final = server_view(fake_langfuse)[exported[0].id]
    assert final.end_time == hook_module.to_otel_nanoseconds(hook_module.parse_timestamp("2026-08-01T10:39:00.000Z"))
    assert final.output == {"role": "assistant", "content": "Batch done."}


def test_late_agent_nests_under_the_agent_that_launched_it(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    view = server_view(fake_langfuse)
    spans = agent_spans(view)
    child2 = view[spans["child2"][0]]
    assert parent_id(child2) == spans["orch"][0]
    assert child2.trace_id == view[spans["orch"][0]].trace_id


def test_quiet_firing_re_emits_nothing(hook_module, fake_langfuse, tmp_path):
    transcript = run_session(hook_module, fake_langfuse, tmp_path)
    before = len(fake_langfuse.observations)
    fire(hook_module, fake_langfuse, transcript, end=True)
    assert len(fake_langfuse.observations) == before


def test_registry_lives_in_its_own_session_file(hook_module, fake_langfuse, tmp_path, isolated_hook_state):
    run_session(hook_module, fake_langfuse, tmp_path)
    registries = list(isolated_hook_state.glob("langfuse_agents.*.json"))
    assert len(registries) == 1
    entries = json.loads(registries[0].read_text(encoding="utf-8"))["agents"]
    assert set(entries) == set(AGENTS)
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text(encoding="utf-8"))
    assert all("agents" not in v for v in state.values())


# ------------------------------------------------------- negative controls
# Each one switches one part of the fix off and shows the fixture then
# reproduces the measured defect, so the assertions above can fail.

def test_control_without_refresh_late_work_is_lost(hook_module, fake_langfuse, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "refresh_grown_agents", lambda *a, **k: 0)
    run_session(hook_module, fake_langfuse, tmp_path)
    view = server_view(fake_langfuse)
    spans = agent_spans(view)
    assert "child2" not in spans, "an agent launched after the interim notification arrived anyway"
    orch = emitted_usage(view, "orch")
    assert orch["messages"] < transcript_usage(tmp_path, "orch")["messages"]


def test_control_without_echo_filter_zero_usage_generations_appear(hook_module, fake_langfuse, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "is_request_echo_row", lambda row: False)
    real_role = hook_module.get_user_or_assistant_role_from_row

    def old_role(row):
        role = real_role(row)
        if role is None and isinstance(row.get("message"), dict):
            # The pre-fix fallback: any row with a nested message.role.
            nested = row["message"].get("role")
            return nested if nested in ("user", "assistant") else None
        return role

    monkeypatch.setattr(hook_module, "get_user_or_assistant_role_from_row", old_role)
    run_session(hook_module, fake_langfuse, tmp_path)
    view = server_view(fake_langfuse)
    assert sum(emitted_usage(view, a)["zero"] for a in AGENTS) > 0


def test_control_without_deterministic_ids_a_resumed_agent_duplicates(hook_module, fake_langfuse, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "DETERMINISTIC_IDS", False)
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(server_view(fake_langfuse))
    assert len(spans["orch"]) > 1


def test_control_without_map_passthrough_workflow_children_are_lost(hook_module, fake_langfuse, tmp_path, monkeypatch):
    real = hook_module.emit_subagent_observations

    def without_maps(*args, **kwargs):
        kwargs["workflow_agent_transcripts_by_run_id"] = None
        if kwargs.get("span_name", "") and str(kwargs.get("span_name")).startswith("Workflow agent"):
            kwargs["subagent_transcripts_by_tool_use_id"] = None
        return real(*args, **kwargs)

    monkeypatch.setattr(hook_module, "emit_subagent_observations", without_maps)
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(server_view(fake_langfuse))
    assert "w1" not in spans and "wchild" not in spans


# ------------------------------------------------------------ the backfill

def test_backfill_is_idempotent_and_complete(hook_module, tmp_path, monkeypatch):
    from conftest import FakeLangfuse

    apply_phase("phase1", tmp_path)
    apply_phase("phase2", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", "backfill-test")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)

    first, second = FakeLangfuse(), FakeLangfuse()
    hook_module.backfill_session(first, config, SESSION, transcript)
    hook_module.backfill_session(second, config, SESSION, transcript)
    first_view, second_view = server_view(first), server_view(second)
    assert set(first_view) == set(second_view), "a re-run must hit the same observation ids"
    assert {o.trace_id for o in first_view.values()} == {o.trace_id for o in second_view.values()}
    for agent in AGENTS:
        assert len(agent_spans(first_view).get(agent, [])) == 1, agent
        assert emitted_usage(first_view, agent)["messages"] == transcript_usage(tmp_path, agent)["messages"], agent


def test_backfill_traces_never_share_ids_with_live_traces(hook_module, fake_langfuse, tmp_path, monkeypatch):
    from conftest import FakeLangfuse

    transcript = run_session(hook_module, fake_langfuse, tmp_path)
    live_traces = {o.trace_id for o in fake_langfuse.observations}
    monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", "backfill-test")
    backfill = FakeLangfuse()
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    hook_module.backfill_session(backfill, config, SESSION, transcript)
    assert live_traces.isdisjoint({o.trace_id for o in backfill.observations})
    keep = hook_module.get_backfill_trace_ids(backfill, SESSION, transcript)
    assert {o.trace_id for o in backfill.observations} <= keep
