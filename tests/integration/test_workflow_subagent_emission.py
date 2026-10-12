from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def copy_workflow_fixture(fixture_transcript_path: Any, tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Copy the workflow fixture's transcript directory (agent transcripts,
    metas, journal) to tmp_path and return the parent rows for incremental
    appends — the fixture file itself must stay untouched so the offset-based
    incremental reader can be driven per firing."""
    source_transcript = fixture_transcript_path("workflow_subagents")
    shutil.copytree(source_transcript.with_suffix(""), tmp_path / "transcript")
    rows = [
        json.loads(line)
        for line in source_transcript.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tmp_path / "transcript.jsonl", rows


def append_rows(transcript: Path, rows: list[dict[str, Any]]) -> None:
    with transcript.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def ts_ns(hook_module: Any, iso_timestamp: str) -> int:
    return hook_module.to_otel_nanoseconds(hook_module.parse_timestamp(iso_timestamp))


def record_scanned_run_dirs(hook_module: Any, monkeypatch: Any) -> list[str]:
    """Record the run directories that an emission reads from disk.

    The returned list stays live. Clear it between firings to limit an assertion
    to one firing."""
    scanned: list[str] = []
    real_scan = hook_module.get_workflow_agents_in_run_dir

    def recording_scan(run_dir: Path) -> Any:
        scanned.append(run_dir.name)
        return real_scan(run_dir)

    monkeypatch.setattr(hook_module, "get_workflow_agents_in_run_dir", recording_scan)
    return scanned


def plain_turn_rows(index: int, minute: int) -> list[dict[str, Any]]:
    """An ordinary turn of two rows. The session continues after a workflow."""
    return [
        {"type": "user", "timestamp": f"2026-07-23T10:{minute:02d}:00.000Z",
         "sessionId": "session-workflow", "uuid": f"user-plain-{index}", "cwd": "/repo",
         "gitBranch": "main", "origin": {"kind": "human"}, "parentUuid": None,
         "permissionMode": "default", "promptId": f"prompt-plain-{index}",
         "promptSource": "sdk", "entrypoint": "cli", "userType": "external",
         "version": "2.1.215", "isSidechain": False,
         "message": {"role": "user", "content": f"Plain question {index}?"}},
        {"type": "assistant", "timestamp": f"2026-07-23T10:{minute:02d}:20.000Z",
         "sessionId": "session-workflow", "uuid": f"assistant-plain-{index}",
         "parentUuid": f"user-plain-{index}", "requestId": f"req-plain-{index}",
         "entrypoint": "cli", "userType": "external", "version": "2.1.215",
         "isSidechain": False,
         "message": {"id": f"msg-plain-{index}", "type": "message", "role": "assistant",
                     "model": "claude-test",
                     "content": [{"type": "text", "text": f"Plain answer {index}."}],
                     "stop_reason": "end_turn",
                     "usage": {"input_tokens": 5, "output_tokens": 5}}},
    ]


# The last workflow-agent activity on disk (r2's trailing StructuredOutput
# tool_result row): the only timestamp that can reach the tool/root span ends
# via workflow_end_timestamp.
LAST_AGENT_ACTIVITY = "2026-07-23T10:01:55.000Z"

# Output-token arithmetic of the fixture's agent generations: r1 is two whole
# messages (210 + 340); r2's first message is SPLIT across two rows sharing
# message.id and the merge keeps the LAST row's usage (tool_use row: 316 — the
# text row's 2 output tokens are dropped), then its retry message adds 273.
# 210 + 340 + 316 + 273 = 1139.
TOTAL_AGENT_OUTPUT_TOKENS = 210 + 340 + 316 + 273


def test_workflow_turn_defers_and_emits_agents_nested_under_the_tool_span(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    def roots():
        return [o for o in fake_langfuse.observations if o.name == "Conversational Turn"]

    # Firing 1: launch ack + Claude's end_turn hand-back (the real transcript
    # shape) — the workflow has not notified, so the turn provably continues
    # and nothing may be exported yet (the root is immutable).
    append_rows(transcript, rows[:4])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow", transcript)
    assert fake_langfuse.observations == []

    # Firing 2: notification delivered + Claude's summary — everything ships.
    append_rows(transcript, rows[4:])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow", transcript)

    names = [o.name for o in fake_langfuse.observations]
    assert len(roots()) == 1
    assert "Tool: Workflow" in names
    assert "Workflow agent: verify-claims/r1" in names
    assert "Workflow agent: verify-claims/r2" in names

    tool_span = next(o for o in fake_langfuse.observations if o.name == "Tool: Workflow")
    assert tool_span._otel_span.parent is roots()[0]._otel_span
    assert tool_span.kwargs["metadata"]["workflow_run_id"] == "wf_test001"
    assert tool_span.kwargs["metadata"]["workflow_name"] == "verify-claims"
    assert tool_span.kwargs["metadata"]["workflow_agent_count"] == 2

    # Both agent spans nest under the launching "Tool: Workflow" span and
    # carry the journal's per-agent result in their metadata.
    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    assert len(agent_spans) == 2
    assert all(o._otel_span.parent is tool_span._otel_span for o in agent_spans)
    r1_span = next(o for o in agent_spans if o.name.endswith("/r1"))
    assert r1_span.kwargs["metadata"]["workflow_agent_id"] == "r1"
    assert json.loads(r1_span.kwargs["metadata"]["workflow_agent_result"]) == {
        "verdict": "claim A holds",
        "confidence": "high",
    }

    # The tool span's exported interval must contain its agent children —
    # spans are immutable once exported, so the end must already cover the
    # agents' last activity, not just the seconds-long launch ack.
    assert tool_span.end_time == ts_ns(hook_module, LAST_AGENT_ACTIVITY)
    assert all(o._otel_span.start_time >= tool_span._otel_span.start_time for o in agent_spans)
    assert all(o.end_time <= tool_span.end_time for o in agent_spans)

    # Generations nest under their agent spans and share the plain "LLM Call"
    # name (context lives on the parent); token math: TOTAL_AGENT_OUTPUT_TOKENS.
    agent_otel_spans = {o._otel_span for o in agent_spans}
    agent_generations = [
        o for o in fake_langfuse.observations
        if o.as_type == "generation" and o._otel_span.parent in agent_otel_spans
    ]
    assert len(agent_generations) == 4
    assert all(o.name == "LLM Call" for o in agent_generations)
    assert sum(g.kwargs["usage_details"]["output"] for g in agent_generations) == TOTAL_AGENT_OUTPUT_TOKENS

    observation_count_after_firing_2 = len(fake_langfuse.observations)

    # SessionEnd: the emission cursor neither duplicates nor starves anything.
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-workflow", transcript, flush_deferred_agent_turns=True
    )
    assert len(fake_langfuse.observations) == observation_count_after_firing_2
    assert len(roots()) == 1
    assert roots()[0].output == {
        "role": "assistant",
        "content": "Both workflow agents completed; claims A and B hold.",
    }


def test_session_end_flush_emits_never_completed_workflow_turn_with_agents(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    # Only the launch + end_turn hand-back reach the transcript; the
    # notification never arrives (killed session). Stop holds the turn ...
    append_rows(transcript, rows[:4])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-dead", transcript)
    assert fake_langfuse.observations == []

    # ... and SessionEnd must still flush it, including whatever workflow
    # agent transcripts exist on disk at that point.
    emitted = hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-workflow-dead", transcript, flush_deferred_agent_turns=True
    )

    assert emitted == 1
    names = [o.name for o in fake_langfuse.observations]
    assert names.count("Conversational Turn") == 1
    assert "Tool: Workflow" in names
    assert "Workflow agent: verify-claims/r1" in names
    assert "Workflow agent: verify-claims/r2" in names
    tool_span = next(o for o in fake_langfuse.observations if o.name == "Tool: Workflow")
    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    assert all(o._otel_span.parent is tool_span._otel_span for o in agent_spans)

    # The agents' end timestamps must propagate through workflow_end_timestamp
    # to both the tool span and the root: every parent-turn row predates the
    # agents here (last assistant 10:00:15, launch ack 10:00:08), so only the
    # r2 generation end can produce these values — dropping
    # workflow_end_timestamp from either fold would truncate the trace to the
    # launch ack and no other fixture timestamp could mask it.
    root_span = next(o for o in fake_langfuse.observations if o.name == "Conversational Turn")
    assert tool_span.end_time == ts_ns(hook_module, LAST_AGENT_ACTIVITY)
    assert root_span.end_time == ts_ns(hook_module, LAST_AGENT_ACTIVITY)


def test_structured_output_ending_agent_falls_back_to_journal_result_output(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    """r2 ends the real way: a message.id-split final message (text row +
    StructuredOutput tool_use row), a schema-error retry with a new
    message.id, and a trailing tool_result row with toolEndsTurn:true."""
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-so", transcript)

    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    r1_span = next(o for o in agent_spans if o.name.endswith("/r1"))
    r2_span = next(o for o in agent_spans if o.name.endswith("/r2"))

    # r2's StructuredOutput tool calls render as tool spans nested under its
    # agent span (both the schema-error attempt and the accepted retry).
    structured_output_spans = [
        o for o in fake_langfuse.observations if o.name == "Tool: StructuredOutput"
    ]
    assert len(structured_output_spans) == 2
    assert all(o._otel_span.parent is r2_span._otel_span for o in structured_output_spans)

    # D1: r2's final message is tool_use-only, so its text extracts empty; the
    # span output falls back to the agent's journal result as a JSON string.
    expected_r2_output = json.dumps(
        {"verdict": "claim B holds", "confidence": "high"}, ensure_ascii=False
    )
    assert r2_span.output == {"role": "assistant", "content": expected_r2_output}
    assert json.loads(r2_span.output["content"]) == {"verdict": "claim B holds", "confidence": "high"}

    # A text-ending agent (r1) keeps its extracted text output — the fallback
    # only fills in when extraction comes back empty.
    assert r1_span.output == {"role": "assistant", "content": "Claim A holds."}

    # Generation split/merge arithmetic stays documented: r1 = 2 whole
    # messages, r2 = 2 merged messages (split rows share message.id).
    agent_otel_spans = {o._otel_span for o in agent_spans}
    agent_generations = [
        o for o in fake_langfuse.observations
        if o.as_type == "generation" and o._otel_span.parent in agent_otel_spans
    ]
    assert len(agent_generations) == 4
    assert sum(g.kwargs["usage_details"]["output"] for g in agent_generations) == TOTAL_AGENT_OUTPUT_TOKENS
    # The r2 tool_use rows carry the real usage shape incl. usage.speed, which
    # surfaces as generation metadata.
    assert any(g.kwargs["metadata"].get("speed") == "standard" for g in agent_generations)


def test_second_workflow_run_on_disk_stays_out_of_the_launching_turn(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    # A second, unrelated run directory exists on disk (e.g. an earlier or
    # later /workflows invocation in the same session) with one finished agent
    # whose activity postdates wf_test001's.
    other_run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_test002"
    other_run_dir.mkdir(parents=True)
    (other_run_dir / "agent-x9.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
    )
    append_rows(other_run_dir / "agent-x9.jsonl", [
        {"type": "user", "timestamp": "2026-07-23T10:03:00.000Z", "sessionId": "session-workflow",
         "agentId": "x9", "uuid": "wf-x9-user-1", "parentUuid": None, "isSidechain": True,
         "message": {"role": "user", "content": "Verify claim X."}},
        {"type": "assistant", "timestamp": "2026-07-23T10:03:30.000Z", "sessionId": "session-workflow",
         "agentId": "x9", "uuid": "wf-x9-assistant-1", "parentUuid": "wf-x9-user-1", "isSidechain": True,
         "message": {"id": "msg-wf-x9-1", "type": "message", "role": "assistant", "model": "claude-test",
                     "content": [{"type": "text", "text": "Claim X holds."}], "stop_reason": "end_turn",
                     "usage": {"input_tokens": 2, "output_tokens": 50}}},
    ])
    append_rows(other_run_dir / "journal.jsonl", [
        {"type": "started", "key": "v2:" + "d4" * 32, "agentId": "x9"},
        {"type": "result", "key": "v2:" + "d4" * 32, "agentId": "x9", "result": {"verdict": "claim X holds"}},
    ])

    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-multi", transcript)

    # Only the launching tool_result's runId (wf_test001) attaches: its two
    # agents nest under the tool span, and nothing from wf_test002 appears in
    # this turn — not as a span, not in any metadata, not in the agent count.
    names = [o.name for o in fake_langfuse.observations]
    assert "Workflow agent: verify-claims/r1" in names
    assert "Workflow agent: verify-claims/r2" in names
    assert not any("x9" in name for name in names)
    assert all(
        o.kwargs.get("metadata", {}).get("workflow_run_id") != "wf_test002"
        for o in fake_langfuse.observations
    )
    tool_span = next(o for o in fake_langfuse.observations if o.name == "Tool: Workflow")
    assert tool_span.kwargs["metadata"]["workflow_agent_count"] == 2
    # x9's later activity (10:03:30) must not leak into the tool span's end.
    assert tool_span.end_time == ts_ns(hook_module, LAST_AGENT_ACTIVITY)


def test_interrupted_agent_emits_no_span_but_stays_in_the_agent_count(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    # Real interrupted-agent shape (harvested): user prompt, the two initial
    # attachment rows, then a user "[Request interrupted by user]" row — the
    # agent never produced a single assistant row.
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_test001"
    (run_dir / "agent-r0.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
    )
    append_rows(run_dir / "agent-r0.jsonl", [
        {"parentUuid": None, "isSidechain": True, "promptId": "prompt-wf-r0-1", "agentId": "r0",
         "type": "user", "message": {"role": "user", "content": "Verify claim C."},
         "uuid": "wf-r0-user-1", "timestamp": "2026-07-23T10:00:14.000Z", "userType": "external",
         "entrypoint": "cli", "cwd": "/repo", "sessionId": "session-workflow",
         "version": "2.1.215", "gitBranch": "main"},
        {"parentUuid": "wf-r0-user-1", "isSidechain": True, "agentId": "r0",
         "attachment": {"type": "deferred_tools_delta", "addedNames": ["WebFetch"],
                        "addedLines": ["WebFetch"], "removedNames": [], "readdedNames": []},
         "type": "attachment", "uuid": "wf-r0-attachment-1", "timestamp": "2026-07-23T10:00:14.001Z",
         "userType": "external", "entrypoint": "cli", "cwd": "/repo",
         "sessionId": "session-workflow", "version": "2.1.215", "gitBranch": "main"},
        {"parentUuid": "wf-r0-attachment-1", "isSidechain": True, "agentId": "r0",
         "attachment": {"type": "skill_listing", "content": "- review: Review a GitHub pull request",
                        "skillCount": 1, "isInitial": True, "names": ["review"]},
         "type": "attachment", "uuid": "wf-r0-attachment-2", "timestamp": "2026-07-23T10:00:14.002Z",
         "userType": "external", "entrypoint": "cli", "cwd": "/repo",
         "sessionId": "session-workflow", "version": "2.1.215", "gitBranch": "main"},
        {"parentUuid": "wf-r0-attachment-2", "isSidechain": True, "promptId": "prompt-wf-r0-1",
         "agentId": "r0", "type": "user",
         "message": {"role": "user", "content": [{"type": "text", "text": "[Request interrupted by user]"}]},
         "uuid": "wf-r0-user-2", "timestamp": "2026-07-23T10:00:20.000Z", "userType": "external",
         "entrypoint": "cli", "cwd": "/repo", "sessionId": "session-workflow",
         "version": "2.1.215", "gitBranch": "main"},
    ])

    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-int", transcript)

    # Decided behavior (D2): an agent whose transcript yields zero turns emits
    # NO span — with no assistant rows there is nothing to backdate a span
    # interval or any generation from, so an empty box would be pure noise.
    # The emission consumes its cursor key and moves on without crashing.
    names = [o.name for o in fake_langfuse.observations]
    assert names.count("Conversational Turn") == 1
    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    assert sorted(o.name for o in agent_spans) == [
        "Workflow agent: verify-claims/r1",
        "Workflow agent: verify-claims/r2",
    ]

    # ... while workflow_agent_count keeps counting every DISCOVERED agent
    # (r0 included). The mismatch is deliberate: the count reports what the
    # run spawned on disk, so a reader can see that an agent existed but left
    # no observable turns — hiding r0 from the count would mask the
    # interruption entirely.
    tool_span = next(o for o in fake_langfuse.observations if o.name == "Tool: Workflow")
    assert tool_span.kwargs["metadata"]["workflow_agent_count"] == 3


def test_truncated_non_json_result_notification_still_resolves_the_turn(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    # Real shape: workflows returning large prose get their <result> cut at
    # ~8k chars with a trailing truncation marker — not JSON, not resumable.
    # The harvested original was 8198 chars; rebuild that length here.
    marker = "\n... (truncated 39258 chars, full result in /repo/tasks/wtest001.output)"
    prose = "VERIFICATION REPORT: claim A cross-checked against three primary sources; claim B confirmed. "
    opaque_result = (prose * 90)[: 8198 - len(marker)] + marker
    assert len(opaque_result) == 8198

    def swap_result(notification_text: str) -> str:
        start = notification_text.index("<result>") + len("<result>")
        end = notification_text.index("</result>")
        return notification_text[:start] + opaque_result + notification_text[end:]

    rows[4]["content"] = swap_result(rows[4]["content"])
    rows[6]["message"]["content"] = swap_result(rows[6]["message"]["content"])

    # The opaque result is carried verbatim as the launch's final_content —
    # nothing ever tries to parse it.
    turns = hook_module.build_turns(rows)
    assert turns[0].tool_results_by_id["toolu_workflow_launch"]["final_content"] == opaque_result

    # Firing 1 (launch + hand-back): held. Firing 2 (notification + summary):
    # the turn still resolves — resolution keys off the notification's
    # tool-use-id, never off the result payload.
    append_rows(transcript, rows[:4])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-trunc", transcript)
    assert fake_langfuse.observations == []

    append_rows(transcript, rows[4:])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-trunc", transcript)

    names = [o.name for o in fake_langfuse.observations]
    assert names.count("Conversational Turn") == 1
    assert "Tool: Workflow" in names

    # The agents are untouched by the opaque notification: their spans emit
    # with the per-agent journal results (metadata and, for r2, the D1 output).
    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    assert len(agent_spans) == 2
    r1_span = next(o for o in agent_spans if o.name.endswith("/r1"))
    assert json.loads(r1_span.kwargs["metadata"]["workflow_agent_result"]) == {
        "verdict": "claim A holds",
        "confidence": "high",
    }
    r2_span = next(o for o in agent_spans if o.name.endswith("/r2"))
    assert json.loads(r2_span.output["content"]) == {"verdict": "claim B holds", "confidence": "high"}


def test_workflow_launch_without_workflow_name_falls_back_to_run_id_labels(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
):
    # Real launches carry toolUseResult.workflowName, but the discovery is
    # tolerant of its absence — span names then fall back to the run id and
    # the metadata omits workflow_name rather than carrying a null.
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    for row in rows:
        tool_use_result = row.get("toolUseResult")
        if isinstance(tool_use_result, dict):
            tool_use_result.pop("workflowName", None)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")

    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-noname", transcript)

    names = [o.name for o in fake_langfuse.observations]
    assert names.count("Conversational Turn") == 1
    assert "Workflow agent: wf_test001/r1" in names
    assert "Workflow agent: wf_test001/r2" in names

    tool_span = next(o for o in fake_langfuse.observations if o.name == "Tool: Workflow")
    assert tool_span.kwargs["metadata"]["workflow_run_id"] == "wf_test001"
    assert "workflow_name" not in tool_span.kwargs["metadata"]
    agent_spans = [o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]
    assert all(o._otel_span.parent is tool_span._otel_span for o in agent_spans)
    assert all("workflow_name" not in o.kwargs["metadata"] for o in agent_spans)


def test_only_the_run_directory_the_turn_references_is_read(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
    monkeypatch,
):

    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    other_run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_test002"
    other_run_dir.mkdir(parents=True)
    (other_run_dir / "agent-x9.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
    )
    (other_run_dir / "agent-x9.jsonl").write_text("", encoding="utf-8")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    scanned = record_scanned_run_dirs(hook_module, monkeypatch)

    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-scope", transcript)

    assert scanned == ["wf_test001"]


def test_closed_workflow_turn_is_not_reread_by_later_firings(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
    monkeypatch,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    scanned = record_scanned_run_dirs(hook_module, monkeypatch)

    # Firing 1 resolves the workflow turn and emits its agents. This reads the
    # run directory.
    append_rows(transcript, rows)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-closed", transcript)
    assert scanned == ["wf_test001"]
    assert len([o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]) == 2

    # The workflow turn is still the open turn. The next turn completes it, and
    # that close walks its rows once more and reads the run again.
    scanned.clear()
    append_rows(transcript, plain_turn_rows(1, 5))
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-closed", transcript)
    assert scanned == ["wf_test001"]

    # A closed turn never enters emission again, so its run directory, whose
    # journal holds every agent's return value, is not read again.
    scanned.clear()
    for index, minute in ((2, 6), (3, 7), (4, 8)):
        append_rows(transcript, plain_turn_rows(index, minute))
        hook_module.emit_new_turns_from_transcript(
            fake_langfuse, config, "session-workflow-closed", transcript
        )

    assert scanned == []
    # No turn is a duplicate, and no turn is lost. The workflow turn and the
    # four ordinary turns each reach Langfuse one time.
    assert len([o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")]) == 2
    assert [o.name for o in fake_langfuse.observations].count("Conversational Turn") == 5


def test_held_workflow_turn_defers_the_run_directory_read_too(
    hook_module,
    fake_langfuse,
    isolated_hook_state,
    fixture_transcript_path,
    tmp_path,
    monkeypatch,
):
    transcript, rows = copy_workflow_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    scanned = record_scanned_run_dirs(hook_module, monkeypatch)

    append_rows(transcript, rows[:4])
    for _ in range(3):
        hook_module.emit_new_turns_from_transcript(
            fake_langfuse, config, "session-workflow-held", transcript
        )
    assert fake_langfuse.observations == []
    assert scanned == []

    # An agent that comes to disk after those firings is still found. The lookup
    # keeps a result for one firing only.
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_test001"
    (run_dir / "agent-r3.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
    )
    append_rows(run_dir / "agent-r3.jsonl", [
        {"type": "user", "timestamp": "2026-07-23T10:01:00.000Z", "sessionId": "session-workflow",
         "agentId": "r3", "uuid": "wf-r3-user-1", "parentUuid": None, "isSidechain": True,
         "message": {"role": "user", "content": "Verify claim C."}},
        {"type": "assistant", "timestamp": "2026-07-23T10:01:30.000Z", "sessionId": "session-workflow",
         "agentId": "r3", "uuid": "wf-r3-assistant-1", "parentUuid": "wf-r3-user-1",
         "isSidechain": True,
         "message": {"id": "msg-wf-r3-1", "type": "message", "role": "assistant",
                     "model": "claude-test",
                     "content": [{"type": "text", "text": "Claim C holds."}],
                     "stop_reason": "end_turn",
                     "usage": {"input_tokens": 2, "output_tokens": 60}}},
    ])

    append_rows(transcript, rows[4:])
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, "session-workflow-held", transcript)

    assert scanned == ["wf_test001"]
    agent_spans = sorted(
        o.name for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")
    )
    assert agent_spans == [
        "Workflow agent: verify-claims/r1",
        "Workflow agent: verify-claims/r2",
        "Workflow agent: verify-claims/r3",
    ]
