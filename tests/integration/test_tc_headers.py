"""Lifting "[tc key=value ...]" prompt headers into agent span metadata.

The tc_headers fixture is sliced from real Claude Code 2.1.x transcripts with
every text body scrubbed: only the header lines survive, with neutral values.
Its tree:

    main turn            [tc migration=m-main work=harness pr= issue=]
    ├── tc-outer         [tc migration=m-test batch=b-test phase=batch]
    │   └── tc-middle    [tc migration=m-test batch=b-test phase=fix]
    │       └── tc-inner (no header: inherits tc-middle's)
    ├── tc-echo          [tc work=refute pr=1 issue=]  (prompt followed by the
    │                    api-request-* echo rows newer Claude Code writes)
    └── tc-plain         (no header: inherits the main turn's)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from conftest import nests_under


def copy_tc_fixture(fixture_transcript_path: Any, tmp_path: Path) -> Path:
    source = fixture_transcript_path("tc_headers")
    shutil.copytree(source.with_suffix(""), tmp_path / "transcript")
    shutil.copy(source, tmp_path / "transcript.jsonl")
    return tmp_path / "transcript.jsonl"


def agent_span(fake_langfuse: Any, agent_id: str) -> Any:
    spans = [
        o for o in fake_langfuse.observations
        if o.name.startswith("Subagent") and o.as_type == "span"
        and str(o.kwargs.get("metadata", {}).get("transcript_path", "")).endswith(f"agent-{agent_id}.jsonl")
    ]
    assert len(spans) == 1, f"expected one span for {agent_id}, got {len(spans)}"
    return spans[0]


def tc_of(span: Any) -> dict:
    return {k: v for k, v in span.kwargs["metadata"].items() if k.startswith("tc_")}


@pytest.fixture
def emitted(hook_module, fake_langfuse, isolated_hook_state, fixture_transcript_path, tmp_path):
    transcript = copy_tc_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-tc", transcript, flush_deferred_agent_turns=True
    )
    return fake_langfuse


# ---- parse_tc_header ----

def test_header_on_the_first_line(hook_module):
    assert hook_module.parse_tc_header("[tc migration=m1 batch=b2 fqn=pkg.a:svc phase=codegen]\nbody") == {
        "migration": "m1", "batch": "b2", "fqn": "pkg.a:svc", "phase": "codegen",
    }


def test_empty_values_are_dropped_not_fatal(hook_module):
    # The "[tc work=<kind> pr=<n> issue=<n>]" convention is often sent with blanks.
    assert hook_module.parse_tc_header("[tc work=harness pr= issue=]\nbody") == {"work": "harness"}
    assert hook_module.parse_tc_header("[tc pr= issue=]\nbody") is None


def test_header_anywhere_in_the_first_line(hook_module):
    assert hook_module.parse_tc_header("Read brief.md (it starts with [tc work=x]) and do it") == {"work": "x"}


def test_indented_header_line_near_the_top(hook_module):
    text = "[Workflow harness — computed task] relayed below:\n  [tc migration=m1 phase=verify]\n  body"
    assert hook_module.parse_tc_header(text) == {"migration": "m1", "phase": "verify"}


def test_mid_line_mention_further_down_is_not_a_header(hook_module):
    text = "Do the work.\nThe convention is to start with [tc migration=x] but this prompt does not.\n"
    assert hook_module.parse_tc_header(text) is None


def test_header_past_the_search_window_is_not_a_header(hook_module):
    text = "intro\n" + "x" * hook_module.TC_HEADER_SEARCH_CHARS + "\n[tc migration=late]\n"
    assert hook_module.parse_tc_header(text) is None


def test_incomplete_header_is_not_a_header(hook_module):
    assert hook_module.parse_tc_header("[tc migration=m1\nbody") is None
    assert hook_module.parse_tc_header("[tc]\nbody") is None


# ---- emission over the real-shaped fixture ----

def test_every_agent_carries_its_own_or_inherited_header(emitted):
    assert tc_of(agent_span(emitted, "tc-outer")) == {
        "tc_source": "own", "tc_migration": "m-test", "tc_batch": "b-test", "tc_phase": "batch",
    }
    assert tc_of(agent_span(emitted, "tc-middle")) == {
        "tc_source": "own", "tc_migration": "m-test", "tc_batch": "b-test", "tc_phase": "fix",
    }
    # Depth 3, no header of its own: its launcher's, whole.
    assert tc_of(agent_span(emitted, "tc-inner")) == {
        "tc_source": "inherited", "tc_migration": "m-test", "tc_batch": "b-test", "tc_phase": "fix",
    }
    # Depth 1, no header: the main turn's.
    assert tc_of(agent_span(emitted, "tc-plain")) == {
        "tc_source": "inherited", "tc_migration": "m-main", "tc_work": "harness",
    }


def test_own_header_replaces_the_inherited_one_whole(emitted):
    # tc-echo runs under a main turn that names migration m-main; its own
    # header names none, and it must not pick m-main up key by key.
    assert tc_of(agent_span(emitted, "tc-echo")) == {"tc_source": "own", "tc_work": "refute", "tc_pr": "1"}


def test_nested_agents_hang_under_their_launchers(emitted):
    outer, middle, inner = (agent_span(emitted, a) for a in ("tc-outer", "tc-middle", "tc-inner"))
    assert nests_under(middle, outer)
    assert nests_under(inner, middle)


def test_prompt_followed_by_request_echo_rows_keeps_its_input(emitted):
    # Claude Code 2.1.301+ writes api-request-blob rows (message.role "user",
    # the injected <system-reminder> context) after the prompt. Read as a chat
    # row, a blob started a second turn, the prompt turn had no reply and was
    # dropped, and the span input became the injected context: no header.
    span = agent_span(emitted, "tc-echo")
    assert span.kwargs["input"]["content"].startswith("[tc work=refute pr=1 issue=]")


def test_main_turn_root_carries_its_header_and_the_session_count(emitted):
    roots = [o for o in emitted.observations if o.name == "Conversational Turn"]
    assert len(roots) == 1
    metadata = roots[0].kwargs["metadata"]
    assert metadata["tc_source"] == "own"
    assert metadata["tc_migration"] == "m-main"
    assert metadata["tc_work"] == "harness"
    assert "tc_pr" not in metadata
    # Five agents: three with their own header, two inheriting, none unattributed.
    assert metadata["tc_session_agents"] == {"agents": 5, "own": 3, "inherited": 2, "unattributed": 0}


def test_headerless_agent_under_a_headerless_turn_is_counted_unattributed(
    hook_module, fake_langfuse, isolated_hook_state, fixture_transcript_path, tmp_path
):
    transcript = copy_tc_fixture(fixture_transcript_path, tmp_path)
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["message"]["content"] = "No header on this prompt."
    transcript.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-tc", transcript, flush_deferred_agent_turns=True
    )
    assert tc_of(agent_span(fake_langfuse, "tc-plain")) == {"tc_source": "none"}
    root = next(o for o in fake_langfuse.observations if o.name == "Conversational Turn")
    assert root.kwargs["metadata"]["tc_source"] == "none"
    assert root.kwargs["metadata"]["tc_session_agents"] == {"agents": 5, "own": 3, "inherited": 1, "unattributed": 1}


def test_trace_tags_name_the_turn_and_direct_agent_headers(hook_module, fixture_transcript_path, tmp_path):
    transcript = copy_tc_fixture(fixture_transcript_path, tmp_path)
    rows = hook_module.read_subagent_jsonl(transcript)
    turn = hook_module.build_turns(rows)[0]
    by_id = hook_module.get_subagent_transcripts_by_tool_use_id(transcript)
    tags = hook_module.get_trace_tags(turn, by_id)
    assert {"tc:migration=m-main", "tc:work=harness", "tc:migration=m-test", "tc:work=refute"} <= set(tags)
    # Per-agent keys stay out of the trace tags.
    assert not any(t.startswith(("tc:batch=", "tc:fqn=", "tc:phase=")) for t in tags)


def test_option_off_lifts_nothing(hook_module, fake_langfuse, isolated_hook_state, fixture_transcript_path, tmp_path, monkeypatch):
    monkeypatch.setattr(hook_module, "TC_HEADERS", False)
    transcript = copy_tc_fixture(fixture_transcript_path, tmp_path)
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-tc", transcript, flush_deferred_agent_turns=True
    )
    for o in fake_langfuse.observations:
        assert not any(k.startswith("tc_") for k in (o.kwargs.get("metadata") or {}))


def test_refresh_keeps_what_an_agent_inherited_at_first_sight(hook_module):
    # A registered agent re-emitted by refresh_grown_agents runs outside its
    # launcher; its stored inheritance, not the empty context, decides.
    registry = hook_module.AgentRegistry(None)
    registry.entries["x"] = {"tc_inherited": {"migration": "m1"}, "tc_source": "inherited"}
    assert hook_module.count_tc_sources(registry.entries) == {"agents": 1, "own": 0, "inherited": 1, "unattributed": 0}
    assert hook_module.resolve_tc(None, registry.entries["x"]["tc_inherited"]) == ({"migration": "m1"}, "inherited")


def test_backfill_dry_run_reports_header_coverage(hook_module, isolated_hook_state, fixture_transcript_path, tmp_path, capsys, monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "CLAUDE_PLUGIN_OPTION_LANGFUSE_PUBLIC_KEY", "CLAUDE_PLUGIN_OPTION_LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    # run_backfill sets these module globals; restore them for later tests.
    monkeypatch.setattr(hook_module, "TRACE_SEED_NAMESPACE", hook_module.TRACE_SEED_NAMESPACE)
    monkeypatch.setattr(hook_module, "BACKFILL_VERIFY_ATTEMPTS", hook_module.BACKFILL_VERIFY_ATTEMPTS)
    # A backfill names the transcript by its session id.
    source = fixture_transcript_path("tc_headers")
    shutil.copytree(source.with_suffix(""), tmp_path / "session-tc")
    transcript = tmp_path / "session-tc.jsonl"
    shutil.copy(source, transcript)
    assert hook_module.run_backfill(["--backfill", str(transcript), "--dry-run", "--force"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["agent_spans"] == 5
    assert summary["agent_tc_sources"] == {"own": 3, "inherited": 2}


# ---- workflow agents ----

def test_workflow_agent_takes_the_computed_task_header_not_the_relayed_request(
    hook_module, fake_langfuse, isolated_hook_state, fixture_transcript_path, tmp_path
):
    source = fixture_transcript_path("workflow_subagents")
    shutil.copytree(source.with_suffix(""), tmp_path / "transcript")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    # The launching main turn carries a header r2 (no header) must inherit.
    rows[0]["message"]["content"] = "[tc migration=m-main work=harness]\nRun the two-agent verification workflow."
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    # r1 gets the real workflow prompt shape: the harness relays the user's
    # request (which itself carries a header), then the computed task.
    r1 = tmp_path / "transcript" / "subagents" / "workflows" / "wf_test001" / "agent-r1.jsonl"
    r1_rows = [json.loads(line) for line in r1.read_text(encoding="utf-8").splitlines() if line.strip()]
    relay = dict(r1_rows[0], uuid="wf-r1-relay", message={"role": "user", "content": (
        "[Workflow harness — user request] The harness relays, verbatim and indented below, the user request:\n"
        "  [tc work=relayed-request]\n  Run the two-agent verification workflow."
    )})
    task = dict(r1_rows[0], parentUuid="wf-r1-relay", message={"role": "user", "content": (
        "[Workflow harness — computed task] The task text below was computed at runtime by a workflow script.\n"
        "  [tc migration=m-wf batch=b-wf fqn=pkg.test:svcW phase=verify]\n  Verify claim A."
    )})
    r1.write_text("".join(json.dumps(r) + "\n" for r in [relay, task, *r1_rows[1:]]), encoding="utf-8")

    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", "user-1")
    hook_module.emit_new_turns_from_transcript(
        fake_langfuse, config, "session-workflow", transcript, flush_deferred_agent_turns=True
    )
    spans = {o.name: o for o in fake_langfuse.observations if o.name.startswith("Workflow agent: ")}
    assert tc_of(spans["Workflow agent: verify-claims/r1"]) == {
        "tc_source": "own", "tc_migration": "m-wf", "tc_batch": "b-wf", "tc_fqn": "pkg.test:svcW", "tc_phase": "verify",
    }
    assert tc_of(spans["Workflow agent: verify-claims/r2"]) == {
        "tc_source": "inherited", "tc_migration": "m-main", "tc_work": "harness",
    }
