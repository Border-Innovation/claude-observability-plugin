"""Nested and resumed agents reach Langfuse whole, ONCE, with their usage.

Fixture (synthetic rows in the Claude Code 2.1.301 shapes):

    main --bg--> orch --bg--> child1 --sync--> grand        (depth 3)
                     \\--Workflow wf_t1 --> w1 --sync--> wchild
                     \\--bg--> child2                       (phase 2 only)

phase1 is the disk at the first Stop: orch has ended a turn while child1
still runs, so main got an interim "completed" notification for orch. phase2
is appended before the second Stop: orch received child1's notification
(as a queued_command attachment, the way a working agent receives it),
launched child2, got a message that opened a new turn, received child2's
notification in that later turn, and finished. Every transcript also
carries api-request / api-request-blob echo rows.

Langfuse v4 reads do not merge two exports of one observation id, so every
assertion here counts RAW exports: an id exported twice is a defect.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from conftest import FakeLangfuse, nests_under


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "resumed_background_agent"
SESSION = "session-resume"
AGENTS = ["orch", "child1", "grand", "child2", "w1", "wchild"]


def apply_phase(phase: str, dst: Path, only: Optional[str] = None) -> None:
    """Append every file of the phase to the same relative path under dst."""
    src_root = FIXTURE / phase
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        if only is not None and only not in src.name:
            continue
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


def by_id(observations: List[Any]) -> Dict[str, Any]:
    return {o.id: o for o in observations}


def parent_id(observation: Any) -> Optional[str]:
    parent = observation._otel_span.parent
    if parent is None:
        return None
    return f"{parent.get_span_context().span_id:016x}"


def agent_name(observation: Any) -> Optional[str]:
    if observation.as_type == "span" and observation.name.startswith(("Subagent", "Workflow agent")):
        path = observation.kwargs.get("metadata", {}).get("transcript_path") or ""
        return path[len("agent-"):-len(".jsonl")]
    return None


def agent_spans(observations: List[Any]) -> Dict[str, List[Any]]:
    spans: Dict[str, List[Any]] = defaultdict(list)
    for observation in observations:
        name = agent_name(observation)
        if name:
            spans[name].append(observation)
    return spans


def agent_of(observation: Any, ids: Dict[str, Any], agent_span_ids: Dict[str, str]) -> str:
    """Walk parent ids; an agent's children may be exported before its span."""
    current = parent_id(observation)
    while current is not None:
        if current in agent_span_ids:
            return agent_span_ids[current]
        parent = ids.get(current)
        current = parent_id(parent) if parent is not None else None
    return "main"


def generations_by_agent(hook_module: Any, observations: List[Any]) -> Dict[str, List[Any]]:
    ids = by_id(observations)
    # Agent span ids are deterministic: derive them so children exported
    # before their agent span still resolve.
    agent_span_ids = {o.id: agent_name(o) for o in observations if agent_name(o)}
    out: Dict[str, List[Any]] = defaultdict(list)
    for o in observations:
        if o.as_type == "generation":
            out[agent_of(o, ids, agent_span_ids)].append(o)
    return out


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


def emitted_usage(generations: List[Any]) -> Dict[str, Any]:
    totals: Counter = Counter()
    zero = 0
    for g in generations:
        usage = g.kwargs.get("usage_details") or {}
        if not usage:
            zero += 1
        totals["output"] += usage.get("output", 0)
        totals["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
    return {"messages": len(generations), "totals": totals, "zero": zero}


def ts_ns(hook_module: Any, iso: str) -> int:
    return hook_module.to_otel_nanoseconds(hook_module.parse_timestamp(iso))


def FIRST_SIGHTING_NS(hook_module: Any) -> int:
    # The launch ack (10:00:03) plus the 1 us display offset.
    return ts_ns(hook_module, "2026-08-01T10:00:03.000Z") + 1000


# ---------------------------------------------------------------- the fix

def test_no_observation_is_ever_exported_twice(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    counts = Counter(o.id for o in fake_langfuse.observations)
    assert [i for i, n in counts.items() if n > 1] == []


def test_every_agent_at_every_depth_arrives_exactly_once(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(fake_langfuse.observations)
    for agent in AGENTS:
        assert len(spans.get(agent, [])) == 1, f"{agent}: {spans.get(agent)}"


def test_each_agent_carries_every_message_once_with_its_final_usage(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    generations = generations_by_agent(hook_module, fake_langfuse.observations)
    for agent in AGENTS:
        expected = transcript_usage(tmp_path, agent)
        got = emitted_usage(generations[agent])
        assert got["zero"] == 0, f"{agent}: zero-usage generations"
        assert got["messages"] == expected["messages"], agent
        assert got["totals"] == expected["totals"], agent


def test_resumed_agent_span_ships_once_after_it_settles(hook_module, fake_langfuse, tmp_path):
    apply_phase("phase1", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    fire(hook_module, fake_langfuse, transcript)
    # Interim: orch still waits on child1, so its span must not ship yet.
    assert "orch" not in agent_spans(fake_langfuse.observations)
    apply_phase("phase2", tmp_path)
    fire(hook_module, fake_langfuse, transcript)
    # Settled now (both children's notifications arrived, the last one in a
    # later turn): the span ships at this Stop, not only at SessionEnd.
    assert len(agent_spans(fake_langfuse.observations).get("orch", [])) == 1
    fire(hook_module, fake_langfuse, transcript, end=True)
    [orch] = agent_spans(fake_langfuse.observations)["orch"]
    assert orch.end_time == ts_ns(hook_module, "2026-08-01T10:39:00.000Z")
    assert orch.output == {"role": "assistant", "content": "Batch done."}
    # Start: where main first showed it (just after the launch ack), kept
    # through the refresh that finalized it - not the agent's own first row.
    assert orch._otel_span.start_time == FIRST_SIGHTING_NS(hook_module)


def test_children_exported_before_their_agent_span_point_at_it(hook_module, fake_langfuse, tmp_path):
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(fake_langfuse.observations)
    orch = spans["orch"][0]
    child2 = spans["child2"][0]
    assert nests_under(child2, orch)
    assert child2.trace_id == orch.trace_id
    # Every generation of orch points at orch's (deterministic) span id.
    gens = generations_by_agent(hook_module, fake_langfuse.observations)["orch"]
    assert gens and all(nests_under(g, orch) for g in gens)


def test_quiet_firing_exports_nothing(hook_module, fake_langfuse, tmp_path):
    transcript = run_session(hook_module, fake_langfuse, tmp_path)
    before = len(fake_langfuse.observations)
    fire(hook_module, fake_langfuse, transcript)
    fire(hook_module, fake_langfuse, transcript, end=True)
    assert len(fake_langfuse.observations) == before


def test_growing_agent_across_many_firings_exports_each_observation_once(hook_module, fake_langfuse, tmp_path):
    """orch's phase-2 rows arrive one per firing (main unchanged): every
    firing may export only what is new and final; raw exports are counted."""
    apply_phase("phase1", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    fire(hook_module, fake_langfuse, transcript)
    # Everything except orch's own phase-2 rows and main's phase-2 rows.
    for src in sorted((FIXTURE / "phase2").rglob("*")):
        if src.is_file() and "agent-orch" not in src.name and src.name != "transcript.jsonl":
            target = tmp_path / src.relative_to(FIXTURE / "phase2")
            target.parent.mkdir(parents=True, exist_ok=True)
            if src.name.endswith(".meta.json"):
                shutil.copyfile(src, target)
            else:
                with target.open("a", encoding="utf-8") as f:
                    f.write(src.read_text(encoding="utf-8"))
    orch_path = tmp_path / "transcript" / "subagents" / "agent-orch.jsonl"
    orch_rows = (FIXTURE / "phase2" / "transcript" / "subagents" / "agent-orch.jsonl").read_text(encoding="utf-8").splitlines()
    exports_per_firing = []
    for row in orch_rows:
        with orch_path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")
        before = len(fake_langfuse.observations)
        fire(hook_module, fake_langfuse, transcript)
        exports_per_firing.append(len(fake_langfuse.observations) - before)
    apply_phase("phase2", tmp_path, only="transcript.jsonl")
    fire(hook_module, fake_langfuse, transcript, end=True)

    counts = Counter(o.id for o in fake_langfuse.observations)
    assert [i for i, n in counts.items() if n > 1] == []
    generations = generations_by_agent(hook_module, fake_langfuse.observations)
    assert emitted_usage(generations["orch"])["messages"] == transcript_usage(tmp_path, "orch")["messages"]
    # A row that adds nothing final (an echo row) costs no export.
    assert 0 in exports_per_firing


def test_registry_lives_in_its_own_session_file(hook_module, fake_langfuse, tmp_path, isolated_hook_state):
    run_session(hook_module, fake_langfuse, tmp_path)
    registries = list(isolated_hook_state.glob("langfuse_agents.*.json"))
    assert len(registries) == 1
    entries = json.loads(registries[0].read_text(encoding="utf-8"))["agents"]
    assert set(entries) == set(AGENTS)
    assert all(e["span_exported"] for e in entries.values())
    state = json.loads((isolated_hook_state / "langfuse_state.json").read_text(encoding="utf-8"))
    assert all("agents" not in v for v in state.values())


def test_registry_keeps_the_first_parent(hook_module, fake_langfuse, tmp_path):
    apply_phase("phase1", tmp_path)
    agent_path = tmp_path / "transcript" / "subagents" / "agent-grand.jsonl"
    registry = hook_module.AgentRegistry(None)
    first = hook_module.carrier_span("11" * 16, "22" * 8)
    second = hook_module.carrier_span("33" * 16, "44" * 8)
    subagent = {"path": agent_path, "agent_id": "grand", "description": "Review grand"}
    with hook_module.active_agent_registry(registry):
        hook_module.emit_subagent_observations(fake_langfuse, first, subagent, None)
        apply_phase("phase2", tmp_path, only="agent-grand")
        hook_module.emit_subagent_observations(fake_langfuse, second, subagent, None)
    assert (registry.entries["grand"]["trace_id"], registry.entries["grand"]["parent_span_id"]) == ("11" * 16, "22" * 8)
    assert {o.trace_id for o in fake_langfuse.observations} == {"11" * 16}


# ------------------------------------------------------- negative controls
# Each one switches one part of the fix off and shows the fixture then
# reproduces the measured defect, so the assertions above can fail.

def test_control_without_refresh_late_work_is_lost(hook_module, fake_langfuse, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "refresh_grown_agents", lambda *a, **k: 0)
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(fake_langfuse.observations)
    assert "child2" not in spans, "an agent launched after the interim notification arrived anyway"


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
    generations = generations_by_agent(hook_module, fake_langfuse.observations)
    assert sum(emitted_usage(generations[a])["zero"] for a in AGENTS) > 0


def test_control_without_attachment_notifications_the_agent_never_settles(hook_module, fake_langfuse, tmp_path, monkeypatch):
    real_text = hook_module.get_task_notification_text
    monkeypatch.setattr(
        hook_module, "get_task_notification_text",
        lambda row: "" if row.get("type") == "attachment" else real_text(row),
    )
    apply_phase("phase1", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    fire(hook_module, fake_langfuse, transcript)
    apply_phase("phase2", tmp_path)
    fire(hook_module, fake_langfuse, transcript)
    # Not settled: child1/child2 never resolve, so orch's later generations
    # and its span wait for SessionEnd.
    assert "orch" not in agent_spans(fake_langfuse.observations)


def test_control_without_deterministic_ids_late_work_is_lost(hook_module, fake_langfuse, tmp_path, monkeypatch):
    # Without deterministic ids the agent ships one-shot (old behavior).
    monkeypatch.setattr(hook_module, "DETERMINISTIC_IDS", False)
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(fake_langfuse.observations)
    assert "child2" not in spans


def test_control_without_map_passthrough_workflow_children_are_lost(hook_module, fake_langfuse, tmp_path, monkeypatch):
    real = hook_module.emit_subagent_observations

    def without_maps(*args, **kwargs):
        kwargs["workflow_agent_transcripts_by_run_id"] = None
        if str(kwargs.get("span_name") or "").startswith("Workflow agent"):
            kwargs["subagent_transcripts_by_tool_use_id"] = None
        return real(*args, **kwargs)

    monkeypatch.setattr(hook_module, "emit_subagent_observations", without_maps)
    run_session(hook_module, fake_langfuse, tmp_path)
    spans = agent_spans(fake_langfuse.observations)
    assert "w1" not in spans and "wchild" not in spans


def test_control_stored_start_ignored_moves_the_span_start(hook_module, fake_langfuse, tmp_path, monkeypatch):
    real = hook_module.emit_subagent_observations

    def without_stored_start(langfuse, parent, subagent, start, *args, **kwargs):
        registry = hook_module._ACTIVE_AGENT_REGISTRY.get()
        if registry is not None and subagent.get("agent_id") in registry.entries:
            registry.entries[subagent["agent_id"]]["start"] = None
        return real(langfuse, parent, subagent, start, *args, **kwargs)

    monkeypatch.setattr(hook_module, "emit_subagent_observations", without_stored_start)
    run_session(hook_module, fake_langfuse, tmp_path)
    [orch] = agent_spans(fake_langfuse.observations)["orch"]
    assert orch._otel_span.start_time != FIRST_SIGHTING_NS(hook_module)


# ------------------------------------------------------------ the backfill

def test_backfill_is_idempotent_and_complete(hook_module, tmp_path, monkeypatch):
    apply_phase("phase1", tmp_path)
    apply_phase("phase2", tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", "backfill-test")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)

    first, second = FakeLangfuse(), FakeLangfuse()
    hook_module.backfill_session(first, config, SESSION, transcript)
    hook_module.backfill_session(second, config, SESSION, transcript)
    assert [o.id for o in first.observations] == [o.id for o in second.observations]
    assert len({o.id for o in first.observations}) == len(first.observations)
    generations = generations_by_agent(hook_module, first.observations)
    for agent in AGENTS:
        assert len(agent_spans(first.observations).get(agent, [])) == 1, agent
        assert emitted_usage(generations[agent])["messages"] == transcript_usage(tmp_path, agent)["messages"], agent


def test_backfill_traces_never_share_ids_with_live_traces(hook_module, fake_langfuse, tmp_path, monkeypatch):
    transcript = run_session(hook_module, fake_langfuse, tmp_path)
    live_traces = {o.trace_id for o in fake_langfuse.observations}
    monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", "backfill-test")
    backfill = FakeLangfuse()
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    hook_module.backfill_session(backfill, config, SESSION, transcript)
    assert live_traces.isdisjoint({o.trace_id for o in backfill.observations})
    keep = hook_module.get_backfill_trace_ids(backfill, SESSION, transcript)
    assert {o.trace_id for o in backfill.observations} <= keep


def test_sync_agent_ending_on_a_tool_call_ships_when_its_result_arrives(hook_module, fake_langfuse, tmp_path):
    """The parent's sync tool_result proves the agent is done even though
    its own transcript ends on a tool call (so it never looks settled)."""
    session = "session-sync"
    sub = tmp_path / "transcript" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-s1.meta.json").write_text(json.dumps({"description": "Sync s1", "toolUseId": "toolu_s1"}))
    rows_s1 = [
        {"type": "user", "uuid": "s-u1", "timestamp": "2026-08-01T11:00:01.000Z", "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "s-a1", "timestamp": "2026-08-01T11:00:02.000Z",
         "message": {"id": "msg_s1", "role": "assistant", "stop_reason": "tool_use", "usage": {"output_tokens": 3},
                     "content": [{"type": "tool_use", "id": "toolu_so", "name": "StructuredOutput", "input": {}}]}},
    ]
    (sub / "agent-s1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_s1))
    main = [
        {"type": "user", "uuid": "m-u1", "timestamp": "2026-08-01T11:00:00.000Z", "message": {"role": "user", "content": "run"}},
        {"type": "assistant", "uuid": "m-a1", "timestamp": "2026-08-01T11:00:00.500Z",
         "message": {"id": "msg_m1", "role": "assistant", "usage": {"output_tokens": 1},
                     "content": [{"type": "tool_use", "id": "toolu_s1", "name": "Agent", "input": {"description": "Sync s1"}}]}},
        {"type": "user", "uuid": "m-r1", "timestamp": "2026-08-01T11:00:03.000Z", "toolUseResult": {"status": "completed"},
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_s1", "content": "ok"}]}},
        {"type": "assistant", "uuid": "m-a2", "timestamp": "2026-08-01T11:00:04.000Z",
         "message": {"id": "msg_m2", "role": "assistant", "stop_reason": "end_turn", "usage": {"output_tokens": 1},
                     "content": [{"type": "text", "text": "done"}]}},
        {"type": "user", "uuid": "m-u2", "timestamp": "2026-08-01T11:01:00.000Z", "message": {"role": "user", "content": "next"}},
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(r) + "\n" for r in main))
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, session, transcript)
    assert len(agent_spans(fake_langfuse.observations).get("s1", [])) == 1
