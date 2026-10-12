from __future__ import annotations

import errno
import json


def test_discovers_subagent_transcripts_by_parent_tool_use_id(
    hook_module,
    fixture_transcript_path,
):
    transcript = fixture_transcript_path("async_agent_completed")

    subagents = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)

    assert set(subagents) == {"toolu_agent_complete"}
    subagent = subagents["toolu_agent_complete"]
    assert subagent["agent_id"] == "agent-complete"
    assert subagent["agent_type"] == "general-purpose"
    assert subagent["description"] == "Summarize docs"
    assert subagent["path"].name == "agent-agent-complete.jsonl"


def test_discovers_nested_subagent_metadata_seen_in_real_claude_code_transcripts(
    hook_module,
    fixture_transcript_path,
):
    transcript = fixture_transcript_path("nested_subagents")

    subagents = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)

    assert set(subagents) == {"toolu_outer_agent", "toolu_inner_agent"}
    assert subagents["toolu_outer_agent"]["agent_id"] == "outer-agent"
    assert subagents["toolu_inner_agent"]["agent_id"] == "inner-agent"
    assert subagents["toolu_inner_agent"]["agent_type"] == "fork"


def test_discovers_workflow_agent_transcripts_by_run_id(hook_module, fixture_transcript_path):
    transcript = fixture_transcript_path("workflow_subagents")

    lookup = hook_module.WorkflowAgentTranscriptsByRunId(transcript)

    agents = lookup.get("wf_test001")
    # The journal also holds a started-row for "r3" with no result and no
    # meta/jsonl on disk (real partial-journal case: agent still running or
    # torn down early) — discovery is meta-driven, so r3 never surfaces.
    assert [agent["agent_id"] for agent in agents] == ["r1", "r2"]
    assert [agent["path"].name for agent in agents] == ["agent-r1.jsonl", "agent-r2.jsonl"]
    assert all(agent["agent_type"] == "workflow-subagent" for agent in agents)
    # Per-agent return values come from the run's journal.jsonl.
    assert agents[0]["result"] == {"verdict": "claim A holds", "confidence": "high"}
    assert agents[1]["result"] == {"verdict": "claim B holds", "confidence": "high"}


def test_workflow_lookup_keeps_concurrent_runs_apart(hook_module, tmp_path):
    # One firing can emit turns that use more than one run, so each run keeps
    # its own entry.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    workflows_dir = tmp_path / "transcript" / "subagents" / "workflows"
    for run_id, agent_id in (("wf_first", "f1"), ("wf_second", "s1")):
        run_dir = workflows_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / f"agent-{agent_id}.meta.json").write_text(
            json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
        )
        (run_dir / f"agent-{agent_id}.jsonl").write_text("", encoding="utf-8")
        (run_dir / "journal.jsonl").write_text(
            json.dumps({"type": "result", "key": "v2:abc", "agentId": agent_id,
                        "result": {"run": run_id}}) + "\n",
            encoding="utf-8",
        )
    lookup = hook_module.WorkflowAgentTranscriptsByRunId(transcript)

    first = lookup.get("wf_first")
    second = lookup.get("wf_second")

    assert [agent["agent_id"] for agent in first] == ["f1"]
    assert [agent["agent_id"] for agent in second] == ["s1"]
    assert first[0]["result"] == {"run": "wf_first"}
    assert second[0]["result"] == {"run": "wf_second"}
    # A second lookup of each run gives the agents of that run.
    assert lookup.get("wf_first") is first
    assert lookup.get("wf_second") is second


def test_workflow_lookup_finds_a_run_directory_without_the_wf_prefix(hook_module, tmp_path):
    # The run id names the directory, so a name without the earlier "wf_" glob
    # pattern resolves too. This is wider on purpose.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "run_abc"
    run_dir.mkdir(parents=True)
    (run_dir / "agent-n1.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}), encoding="utf-8"
    )
    (run_dir / "agent-n1.jsonl").write_text("", encoding="utf-8")

    agents = hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("run_abc")

    assert [agent["agent_id"] for agent in agents] == ["n1"]


def test_workflow_lookup_memoizes_run_ids_that_have_no_agents(
    hook_module,
    tmp_path,
    monkeypatch,
):
    # A run directory can be absent. The lookup must keep that empty result too,
    # or each observation that uses the run makes a new read.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    scanned: list[str] = []
    monkeypatch.setattr(
        hook_module,
        "get_workflow_agents_in_run_dir",
        lambda run_dir: scanned.append(run_dir.name) or [],
    )
    lookup = hook_module.WorkflowAgentTranscriptsByRunId(transcript)

    assert lookup.get("wf_gone") is None
    assert lookup.get("wf_gone") is None
    assert scanned == ["wf_gone"]


def test_workflow_lookup_rejects_run_ids_that_leave_the_workflows_directory(
    hook_module,
    fixture_transcript_path,
    monkeypatch,
):
    # The lookup makes a path from transcript data, so only a plain directory
    # name must resolve, and a rejected run id must cause no read.
    transcript = fixture_transcript_path("workflow_subagents")
    scanned: list[str] = []
    monkeypatch.setattr(
        hook_module,
        "get_workflow_agents_in_run_dir",
        lambda run_dir: scanned.append(str(run_dir)) or [],
    )
    lookup = hook_module.WorkflowAgentTranscriptsByRunId(transcript)

    for hostile_run_id in ("..", "../wf_test001", "wf_test001/..", "nested/wf_test001",
                           "/etc", ".", "wf_test001/"):
        assert lookup.get(hostile_run_id) is None, hostile_run_id
    # An absent or incorrect run id also gives None, and makes no file read.
    assert lookup.get(None) is None
    assert lookup.get("") is None
    assert lookup.get(123) is None
    assert scanned == []


def test_workflow_agents_stay_invisible_to_classic_subagent_discovery(
    hook_module,
    fixture_transcript_path,
):
    # Documented split: the classic function keys strictly by toolUseId and
    # stays non-recursive. Only the run-id lookup finds workflow agents.
    transcript = fixture_transcript_path("workflow_subagents")

    assert hook_module.get_subagent_transcripts_by_tool_use_id(transcript) == {}


def test_discovers_named_teammate_meta_without_tool_use_id_via_launch_input_name(hook_module, tmp_path):
    # A teammate meta has no toolUseId. Only input.name can attribute it.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_builder",
                        "name": "Agent",
                        "input": {
                            "name": "builder-issue-4",
                            "taskKind": "in_process_teammate",
                            "prompt": "Build the fix",
                        },
                    },
                ],
            },
        }) + "\n",
        encoding="utf-8",
    )
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    # Real teammate file stems are "agent-a<name>-<hex>". The stem never matches a launch.
    (subagent_dir / "agent-abuilder-issue-4-03b96faf.meta.json").write_text(
        json.dumps({
            "agentType": "builder-issue-4",
            "description": "Build issue 4",
            "name": "builder-issue-4",
            "spawnDepth": 0,
            "model": "sonnet",
            "taskKind": "in_process_teammate",
            "teamName": "session-test",
        }),
        encoding="utf-8",
    )
    (subagent_dir / "agent-abuilder-issue-4-03b96faf.jsonl").write_text("", encoding="utf-8")

    subagents = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)

    assert set(subagents) == {"toolu_builder"}
    assert subagents["toolu_builder"]["agent_id"] == "abuilder-issue-4-03b96faf"
    assert subagents["toolu_builder"]["agent_type"] == "builder-issue-4"
    assert subagents["toolu_builder"]["description"] == "Build issue 4"


def test_named_teammate_fallback_ignores_ambiguous_names(hook_module, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_one", "name": "Agent", "input": {"name": "builder"}}
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_two", "name": "Agent", "input": {"name": "builder"}}
                ],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-builder.meta.json").write_text(
        json.dumps({"agentType": "builder", "name": "builder", "taskKind": "in_process_teammate"}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-builder.jsonl").write_text("", encoding="utf-8")

    assert hook_module.get_subagent_transcripts_by_tool_use_id(transcript) == {}


def test_reused_teammate_name_drops_both_transcripts(hook_module, tmp_path):
    # Real teammate launch results carry no per-spawn key (snake_case agent_id is
    # "<name>@<team>"). A reused name is ambiguous, so both metas are skipped.
    rows = [
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_first", "name": "Agent", "input": {"name": "builder"}}]}},
        {"type": "user", "toolUseResult": {"status": "teammate_spawned", "agent_id": "builder@session-x",
                                           "teammate_id": "builder@session-x", "name": "builder"},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_first",
              "content": "Spawned successfully. agent_id: builder@session-x"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_second", "name": "Agent", "input": {"name": "builder"}}]}},
        {"type": "user", "toolUseResult": {"status": "teammate_spawned", "agent_id": "builder@session-x",
                                           "teammate_id": "builder@session-x", "name": "builder"},
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_second",
              "content": "Spawned successfully. agent_id: builder@session-x"}]}},
    ]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    for stem in ("abuilder-11111111", "abuilder-22222222"):
        (subagent_dir / f"agent-{stem}.meta.json").write_text(
            json.dumps({"agentType": "builder", "name": "builder", "taskKind": "in_process_teammate"}),
            encoding="utf-8",
        )
        (subagent_dir / f"agent-{stem}.jsonl").write_text("", encoding="utf-8")

    assert hook_module.get_subagent_transcripts_by_tool_use_id(transcript) == {}


def test_two_metas_claiming_one_launch_drop_both(hook_module, tmp_path):
    # A stale meta and a live one can share a name while only one launch row
    # exists. Attribution is unknowable, so neither meta may win.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_only", "name": "Agent", "input": {"name": "reviewer"}}]}}) + "\n",
        encoding="utf-8",
    )
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    for stem in ("areviewer-11111111", "areviewer-22222222"):
        (subagent_dir / f"agent-{stem}.meta.json").write_text(
            json.dumps({"agentType": "reviewer", "name": "reviewer", "taskKind": "in_process_teammate"}),
            encoding="utf-8",
        )
        (subagent_dir / f"agent-{stem}.jsonl").write_text("", encoding="utf-8")

    assert hook_module.get_subagent_transcripts_by_tool_use_id(transcript) == {}


def test_non_dict_meta_is_skipped_without_crashing(hook_module, tmp_path):
    # A meta holding a JSON array raised AttributeError before the dict guard.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-broken.meta.json").write_text('["not", "a", "dict"]', encoding="utf-8")
    (subagent_dir / "agent-broken.jsonl").write_text("", encoding="utf-8")
    (subagent_dir / "agent-ok.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "toolUseId": "toolu_ok", "spawnDepth": 1}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-ok.jsonl").write_text("", encoding="utf-8")

    assert set(hook_module.get_subagent_transcripts_by_tool_use_id(transcript)) == {"toolu_ok"}


def test_unattributable_teammate_meta_does_not_break_classic_discovery(hook_module, tmp_path):
    # A meta with no match must not stop classic discovery in the same directory.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-classic.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "toolUseId": "toolu_classic", "spawnDepth": 1}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-classic.jsonl").write_text("", encoding="utf-8")
    (subagent_dir / "agent-orphan.meta.json").write_text(
        json.dumps({"agentType": "orphan", "name": "orphan", "taskKind": "in_process_teammate"}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-orphan.jsonl").write_text("", encoding="utf-8")

    subagents = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)

    assert set(subagents) == {"toolu_classic"}


def test_workflow_discovery_ignores_bad_or_foreign_metadata(hook_module, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_x1"
    run_dir.mkdir(parents=True)
    (run_dir / "agent-bad.meta.json").write_text("{not-json", encoding="utf-8")
    (run_dir / "agent-classic.meta.json").write_text(
        json.dumps({"agentType": "general-purpose", "toolUseId": "toolu_x"}),
        encoding="utf-8",
    )
    (run_dir / "agent-classic.jsonl").write_text("", encoding="utf-8")
    (run_dir / "agent-no-jsonl.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}),
        encoding="utf-8",
    )

    assert hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_x1") is None


def test_workflow_discovery_drops_run_dirs_with_journal_but_no_agents(hook_module, tmp_path):
    # A run dir can exist with only its journal (e.g. the workflow crashed
    # before spawning agents, or agents were cleaned up): no agent metas means
    # nothing to emit, so the run must not surface at all.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_journal_only"
    run_dir.mkdir(parents=True)
    (run_dir / "journal.jsonl").write_text(
        json.dumps({"type": "started", "key": "v2:abc", "agentId": "r1"}) + "\n",
        encoding="utf-8",
    )

    assert hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_journal_only") is None


def test_workflow_discovery_ignores_orphan_agent_jsonl_without_meta(hook_module, tmp_path):
    # Discovery is meta-driven: an agent transcript without its meta.json
    # (half-written run, partial cleanup) stays invisible instead of crashing
    # or emitting an unidentifiable agent.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_orphan"
    run_dir.mkdir(parents=True)
    (run_dir / "agent-orphan.jsonl").write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}) + "\n",
        encoding="utf-8",
    )

    assert hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_orphan") is None


def test_workflow_discovery_accepts_metas_with_extra_real_world_keys(hook_module, tmp_path):
    # Harvested ground truth: real metas carry only agentType/spawnDepth, plus
    # worktreePath when the run launched from a worktree (key order
    # agentType -> worktreePath -> spawnDepth, matching the real meta).
    # "model" is NOT harvested from any real meta — it is a hypothetical extra
    # key probing that discovery keys on agentType only and tolerates unknown
    # additions.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_x3"
    run_dir.mkdir(parents=True)
    (run_dir / "agent-a1.meta.json").write_text(
        json.dumps({
            "agentType": "workflow-subagent",
            "worktreePath": "/repo/.claude/worktrees/wf_x3-11",
            "spawnDepth": 1,
            "model": "claude-test",
        }),
        encoding="utf-8",
    )
    (run_dir / "agent-a1.jsonl").write_text("", encoding="utf-8")

    agents = hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_x3")

    assert [agent["agent_id"] for agent in agents] == ["a1"]
    assert agents[0]["agent_type"] == "workflow-subagent"


def test_workflow_discovery_without_journal_leaves_results_empty(hook_module, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    run_dir = tmp_path / "transcript" / "subagents" / "workflows" / "wf_x2"
    run_dir.mkdir(parents=True)
    (run_dir / "agent-a1.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "spawnDepth": 1}),
        encoding="utf-8",
    )
    (run_dir / "agent-a1.jsonl").write_text("", encoding="utf-8")

    agents = hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_x2")

    assert [agent["agent_id"] for agent in agents] == ["a1"]
    assert agents[0]["result"] is None


def test_workflow_lookup_survives_a_run_directory_the_filesystem_rejects(
    hook_module,
    tmp_path,
    monkeypatch,
):
    # The hook runs in the Stop path of Claude Code. Thus a run path that the
    # operating system refuses must not stop the turn.
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")

    def raise_name_too_long(_run_dir):
        raise OSError(errno.ENAMETOOLONG, "File name too long")

    monkeypatch.setattr(hook_module, "get_workflow_agents_in_run_dir", raise_name_too_long)

    assert hook_module.WorkflowAgentTranscriptsByRunId(transcript).get("wf_x") is None


def test_ignores_bad_or_incomplete_subagent_metadata(hook_module, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    subagent_dir = tmp_path / "transcript" / "subagents"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "agent-bad.meta.json").write_text("{not-json", encoding="utf-8")
    (subagent_dir / "agent-missing-jsonl.meta.json").write_text(
        json.dumps({"toolUseId": "toolu_missing"}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-missing-tool-id.meta.json").write_text(
        json.dumps({"description": "No tool id"}),
        encoding="utf-8",
    )
    (subagent_dir / "agent-missing-tool-id.jsonl").write_text("", encoding="utf-8")

    assert hook_module.get_subagent_transcripts_by_tool_use_id(transcript) == {}
