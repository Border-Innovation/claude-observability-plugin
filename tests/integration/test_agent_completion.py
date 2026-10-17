"""When an agent (and a waiting turn) counts as finished.

Synthetic transcripts in the Claude Code 2.1.301 shapes. Every test counts
RAW exports: Langfuse v4 reads do not merge two exports of one id.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

SESSION = "session-completion"


def ts(minute: int, second: int = 0, hour: int = 10) -> str:
    return f"2026-08-02T{hour:02d}:{minute:02d}:{second:02d}.000Z"


def user(uuid: str, t: str, content: Any, **extra: Any) -> Dict[str, Any]:
    row = {"type": "user", "uuid": uuid, "timestamp": t, "sessionId": SESSION,
           "message": {"role": "user", "content": content}}
    row.update(extra)
    return row


def asst(uuid: str, t: str, mid: str, content: List[Any], output: int, stop: Any) -> Dict[str, Any]:
    return {"type": "assistant", "uuid": uuid, "timestamp": t, "sessionId": SESSION,
            "message": {"id": mid, "role": "assistant", "model": "claude-test", "content": content,
                        "stop_reason": stop, "usage": {"input_tokens": 1, "output_tokens": output}}}


def agent_launch(tid: str, description: str) -> Dict[str, Any]:
    return {"type": "tool_use", "id": tid, "name": "Agent",
            "input": {"description": description, "prompt": "go", "run_in_background": True}}


def launched(uuid: str, t: str, tid: str, agent_id: str) -> Dict[str, Any]:
    return user(uuid, t, [{"type": "tool_result", "tool_use_id": tid,
                           "content": [{"type": "text", "text": "Async agent launched successfully."}]}],
                toolUseResult={"isAsync": True, "status": "async_launched", "agentId": agent_id})


def notification_text(task_id: str, tid: str, result: str) -> str:
    return (f"<task-notification>\n<task-id>{task_id}</task-id>\n<tool-use-id>{tid}</tool-use-id>\n"
            f"<status>completed</status>\n<result>{result}</result>\n</task-notification>")


def write(path: Path, rows: List[Dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def setup_session(tmp_path: Path, agent_rows: List[Dict[str, Any]]) -> Path:
    """main launches agent x in the background; x's final notification
    reaches main, so main's turn emits x at the first firing."""
    sub = tmp_path / "transcript" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-x.meta.json").write_text(json.dumps({"description": "Agent x", "toolUseId": "toolu_x"}))
    write(sub / "agent-x.jsonl", agent_rows)
    (sub / "agent-y.meta.json").write_text(json.dumps({"description": "Agent y", "toolUseId": "toolu_y"}))
    write(sub / "agent-y.jsonl", [
        user("y-u", ts(1, 10), "go"),
        asst("y-a", ts(1, 20), "msg_y", [{"type": "text", "text": "y done"}], 5, "end_turn"),
    ])
    transcript = tmp_path / "transcript.jsonl"
    write(transcript, [
        user("m-u1", ts(0), "start"),
        asst("m-a1", ts(0, 1), "msg_m1", [agent_launch("toolu_x", "Agent x")], 2, "tool_use"),
        launched("m-r1", ts(0, 2), "toolu_x", "x"),
        asst("m-a2", ts(0, 3), "msg_m2", [{"type": "text", "text": "launched"}], 2, "end_turn"),
        user("m-n1", ts(3), notification_text("x", "toolu_x", "x said done"), origin={"kind": "task-notification"}),
        user("m-u2", ts(4), "next"),
        asst("m-a3", ts(4, 1), "msg_m3", [{"type": "text", "text": "ok"}], 2, "end_turn"),
    ])
    return transcript


def fire(hook_module: Any, fake_langfuse: Any, transcript: Path, *, end: bool = False) -> None:
    config = hook_module.LangfuseConfig("public", "secret", "https://example.test", None)
    hook_module.emit_new_turns_from_transcript(fake_langfuse, config, SESSION, transcript,
                                               flush_deferred_agent_turns=end)


def spans_of(fake_langfuse: Any, description: str) -> List[Any]:
    return [o for o in fake_langfuse.observations if o.name == f"Subagent: {description}"]


def generation_outputs(fake_langfuse: Any) -> List[int]:
    return sorted(int((o.kwargs.get("usage_details") or {}).get("output", 0))
                  for o in fake_langfuse.observations if o.as_type == "generation")


def assert_no_id_twice(fake_langfuse: Any) -> None:
    counts = Counter(o.id for o in fake_langfuse.observations)
    assert [i for i, n in counts.items() if n > 1] == []


# x launches y in the background; y's notification reaches x as an isMeta
# user row, the way a subagent receives it.
X_WITH_META_NOTIFICATION = [
    user("x-u", ts(0, 5), "go"),
    asst("x-a1", ts(1), "msg_x1", [agent_launch("toolu_y", "Agent y")], 3, "tool_use"),
    launched("x-r1", ts(1, 1), "toolu_y", "y"),
    asst("x-a2", ts(1, 2), "msg_x2", [{"type": "text", "text": "waiting"}], 3, "end_turn"),
    user("x-n1", ts(2), "[SYSTEM NOTIFICATION - NOT USER INPUT]\n" + notification_text("y", "toolu_y", "y done"),
         isMeta=True, origin={"kind": "task-notification"}),
    asst("x-a3", ts(2, 30), "msg_x3", [{"type": "text", "text": "x done"}], 7, "end_turn"),
]


def test_meta_notification_resolves_the_launch_and_the_agent_settles(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, X_WITH_META_NOTIFICATION)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1, "x did not settle at the Stop"
    # x's three generations ship at the Stop, not only at SessionEnd.
    assert [3, 3, 7] == sorted(o for o in generation_outputs(fake_langfuse) if o in (3, 7))
    assert_no_id_twice(fake_langfuse)


def test_last_message_still_being_written_is_held(hook_module, fake_langfuse, tmp_path):
    rows = [
        user("x-u", ts(0, 5), "go"),
        asst("x-a1", ts(1), "msg_x1", [{"type": "text", "text": "step"}], 9, "end_turn"),
        # Only the first part of msg_x2 is on disk: partial usage, no stop_reason.
        asst("x-a2a", ts(1, 5), "msg_x2", [{"type": "thinking", "thinking": "hm", "signature": "s"}], 4, None),
    ]
    transcript = setup_session(tmp_path, rows)
    fire(hook_module, fake_langfuse, transcript)
    assert spans_of(fake_langfuse, "Agent x") == [], "a half-written agent settled"
    assert 4 not in generation_outputs(fake_langfuse), "a partial generation shipped"
    write(tmp_path / "transcript" / "subagents" / "agent-x.jsonl",
          [asst("x-a2b", ts(1, 9), "msg_x2", [{"type": "text", "text": "x done"}], 336, "end_turn")], mode="a")
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1
    assert generation_outputs(fake_langfuse).count(336) == 1
    assert 4 not in generation_outputs(fake_langfuse)
    assert_no_id_twice(fake_langfuse)


UNRESOLVED = [
    user("x-u", ts(0, 5), "go"),
    asst("x-a1", ts(1), "msg_x1", [agent_launch("toolu_y", "Agent y")], 3, "tool_use"),
    launched("x-r1", ts(1, 1), "toolu_y", "y"),
    # Written after the launch whose result never arrives: mutable.
    asst("x-a2", ts(1, 2), "msg_x2", [{"type": "text", "text": "waiting"}], 11, "end_turn"),
]


def test_running_agent_ships_only_what_cannot_change_then_sessionend_finalizes(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, UNRESOLVED)
    fire(hook_module, fake_langfuse, transcript)
    assert spans_of(fake_langfuse, "Agent x") == []
    assert 11 not in generation_outputs(fake_langfuse), "a generation after an unresolved launch shipped"
    assert 3 in generation_outputs(fake_langfuse), "the launch generation itself is final and ships"
    fire(hook_module, fake_langfuse, transcript, end=True)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1, "SessionEnd did not finalize the unsettled agent"
    assert generation_outputs(fake_langfuse).count(11) == 1
    assert_no_id_twice(fake_langfuse)


# ---------------------------------------------------- waiting main turns

def workflow_turn_rows(stop_result: Any, stop_hours_later: int = 0) -> List[Dict[str, Any]]:
    stop_row = user("m-r2", ts(6, 2, 10 + stop_hours_later), [{"type": "tool_result", "tool_use_id": "toolu_stop",
                                                              "content": json.dumps(stop_result)}])
    if isinstance(stop_result, dict):
        stop_row["toolUseResult"] = stop_result
    else:
        stop_row["toolUseResult"] = stop_result
        stop_row["message"]["content"][0]["is_error"] = True
    return [
        user("m-u1", ts(0), "start a workflow"),
        asst("m-a1", ts(0, 1), "msg_m1", [{"type": "tool_use", "id": "toolu_wf", "name": "Workflow",
                                           "input": {"script": "round"}}], 2, "tool_use"),
        user("m-r1", ts(0, 2), [{"type": "tool_result", "tool_use_id": "toolu_wf", "content": "Workflow launched."}],
             toolUseResult={"status": "async_launched", "isAsync": True, "taskId": "wtask1",
                            "taskType": "local_workflow", "runId": "wf_none"}),
        asst("m-a2", ts(0, 3), "msg_m2", [{"type": "text", "text": "running"}], 2, "end_turn"),
        user("m-u2", ts(6, 0, 10 + stop_hours_later), "stop it"),
        asst("m-a3", ts(6, 1, 10 + stop_hours_later), "msg_m3",
             [{"type": "tool_use", "id": "toolu_stop", "name": "TaskStop", "input": {"task_id": "wtask1"}}], 2, "tool_use"),
        stop_row,
        asst("m-a4", ts(6, 3, 10 + stop_hours_later), "msg_m4", [{"type": "text", "text": "stopped"}], 2, "end_turn"),
        user("m-u3", ts(7, 0, 10 + stop_hours_later), "thanks"),
    ]


def roots_by_user_uuid(fake_langfuse: Any) -> List[str]:
    return [o.kwargs["metadata"].get("user_row_uuid") or o.kwargs["input"]["content"]
            for o in fake_langfuse.observations if o.name == "Conversational Turn"]


def test_taskstop_resolves_a_launch_that_will_never_notify(hook_module, fake_langfuse, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    write(transcript, workflow_turn_rows({"message": "Successfully stopped task: wtask1", "task_id": "wtask1",
                                          "task_type": "local_workflow"}))
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" in roots_by_user_uuid(fake_langfuse)


def test_control_failed_taskstop_keeps_the_turn_waiting(hook_module, fake_langfuse, tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    write(transcript, workflow_turn_rows("Error: Task wtask1 is not running (status: failed)"))
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" not in roots_by_user_uuid(fake_langfuse)


def test_a_turn_waiting_past_the_bound_ships(hook_module, fake_langfuse, tmp_path):
    # The failed stop leaves the launch unresolved; 13 h later it ships anyway.
    transcript = tmp_path / "transcript.jsonl"
    write(transcript, workflow_turn_rows("Error: not running", stop_hours_later=13))
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" in roots_by_user_uuid(fake_langfuse)


def _age(path: Path, seconds: float) -> None:
    import os
    import time
    then = time.time() - seconds
    os.utime(path, (then, then))


def test_agent_idle_past_the_bound_settles_despite_a_launch_that_never_notifies(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, UNRESOLVED)
    _age(tmp_path / "transcript" / "subagents" / "agent-x.jsonl", hook_module.MAX_DEFER_SECONDS + 60)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1
    assert generation_outputs(fake_langfuse).count(11) == 1


def test_agent_killed_mid_message_settles_after_the_idle_bound(hook_module, fake_langfuse, tmp_path):
    rows = [
        user("x-u", ts(0, 5), "go"),
        asst("x-a1", ts(1), "msg_x1", [{"type": "thinking", "thinking": "hm", "signature": "s"}], 4, None),
    ]
    transcript = setup_session(tmp_path, rows)
    _age(tmp_path / "transcript" / "subagents" / "agent-x.jsonl", hook_module.AGENT_IDLE_SETTLE_SECONDS + 60)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1


# ------------------------------------------------ revisits between Stops

SETTLED_X = [
    user("x-u", ts(0, 5), "go"),
    asst("x-a1", ts(1), "msg_x1", [{"type": "text", "text": "x done"}], 5, "end_turn"),
]


def test_woken_agents_final_message_ships_even_when_never_given_a_stop_reason(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, SETTLED_X)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1
    # Woken by a notification; its new final message is written with
    # stop_reason null and never completed.
    write(tmp_path / "transcript" / "subagents" / "agent-x.jsonl", [
        user("x-n", ts(5), "[SYSTEM NOTIFICATION - NOT USER INPUT]\n" + notification_text("z", "toolu_z", "late"),
             isMeta=True, origin={"kind": "task-notification"}),
        asst("x-a2", ts(5, 10), "msg_x2", [{"type": "text", "text": "noted"}], 77, None),
    ], mode="a")
    fire(hook_module, fake_langfuse, transcript)
    assert 77 not in generation_outputs(fake_langfuse), "held message shipped while it may still grow"
    fire(hook_module, fake_langfuse, transcript)
    fire(hook_module, fake_langfuse, transcript, end=True)
    assert generation_outputs(fake_langfuse).count(77) == 1, "the woken agent's last message was lost"
    assert len(spans_of(fake_langfuse, "Agent x")) == 1
    assert_no_id_twice(fake_langfuse)


def test_held_message_ships_after_the_idle_bound_without_sessionend(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, SETTLED_X)
    fire(hook_module, fake_langfuse, transcript)
    path = tmp_path / "transcript" / "subagents" / "agent-x.jsonl"
    write(path, [asst("x-a2", ts(5, 10), "msg_x2", [{"type": "text", "text": "noted"}], 77, None)], mode="a")
    fire(hook_module, fake_langfuse, transcript)  # first seen fresh: held
    assert 77 not in generation_outputs(fake_langfuse)
    _age(path, hook_module.AGENT_IDLE_SETTLE_SECONDS + 60)  # then it ages, unchanged
    fire(hook_module, fake_langfuse, transcript)
    assert generation_outputs(fake_langfuse).count(77) == 1


def test_unresolved_agent_seen_fresh_settles_once_it_ages(hook_module, fake_langfuse, tmp_path):
    transcript = setup_session(tmp_path, UNRESOLVED)
    fire(hook_module, fake_langfuse, transcript)
    assert spans_of(fake_langfuse, "Agent x") == []
    fire(hook_module, fake_langfuse, transcript)
    assert spans_of(fake_langfuse, "Agent x") == []
    _age(tmp_path / "transcript" / "subagents" / "agent-x.jsonl", hook_module.MAX_DEFER_SECONDS + 60)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1


def test_background_shell_keeps_the_agent_awake_until_it_notifies(hook_module, fake_langfuse, tmp_path):
    rows = [
        user("x-u", ts(0, 5), "go"),
        asst("x-a1", ts(1), "msg_x1", [{"type": "tool_use", "id": "toolu_sh", "name": "Bash",
                                        "input": {"command": "sleep 600", "run_in_background": True}}], 3, "tool_use"),
        user("x-r1", ts(1, 1), [{"type": "tool_result", "tool_use_id": "toolu_sh",
                                 "content": "Command running in background with ID: bsh1"}],
             toolUseResult={"stdout": "", "stderr": "", "interrupted": False, "backgroundTaskId": "bsh1"}),
        asst("x-a2", ts(1, 2), "msg_x2", [{"type": "text", "text": "waiting for the shell"}], 3, "end_turn"),
    ]
    transcript = setup_session(tmp_path, rows)
    fire(hook_module, fake_langfuse, transcript)
    assert spans_of(fake_langfuse, "Agent x") == [], "span shipped while the agent's shell still runs"
    write(tmp_path / "transcript" / "subagents" / "agent-x.jsonl", [
        # The shell's notification names only its task id.
        user("x-n", ts(9), "<task-notification>\n<task-id>bsh1</task-id>\n<status>completed</status>\n"
             "<result>shell done</result>\n</task-notification>", origin={"kind": "task-notification"}),
        asst("x-a3", ts(9, 5), "msg_x3", [{"type": "text", "text": "x done"}], 21, "end_turn"),
    ], mode="a")
    fire(hook_module, fake_langfuse, transcript)
    [span] = spans_of(fake_langfuse, "Agent x")
    assert span.output == {"role": "assistant", "content": "x done"}


# ------------------------------------------------------------- TaskStop

def test_control_failed_taskstop_with_a_dict_result_resolves_nothing(hook_module):
    row = user("r", ts(1), [{"type": "tool_result", "tool_use_id": "toolu_stop", "is_error": True,
                             "content": "not running"}],
               toolUseResult={"task_id": "wtask1", "message": "not running"})
    assert hook_module.get_stopped_task_id(row) is None


def taskstop_row(task_id: str = "wtask1") -> Dict[str, Any]:
    return user("m-stop", ts(6, 2), [{"type": "tool_result", "tool_use_id": "toolu_stop",
                                      "content": f"Successfully stopped task: {task_id}"}],
                toolUseResult={"message": f"Successfully stopped task: {task_id}", "task_id": task_id,
                               "task_type": "local_workflow"})


def test_taskstop_expansion_is_idempotent(hook_module):
    launch = user("m-r1", ts(0, 2), [{"type": "tool_result", "tool_use_id": "toolu_wf", "content": "launched"}],
                  toolUseResult={"status": "async_launched", "taskId": "wtask1"})
    once = hook_module.expand_task_stop_rows([launch, taskstop_row()], [launch, taskstop_row()])
    twice = hook_module.expand_task_stop_rows(once, once)
    assert len(once) == 3 and len(twice) == 3


def test_taskstop_falls_back_to_the_task_id_map(hook_module):
    rows = [taskstop_row("agent7")]
    expanded = hook_module.expand_task_stop_rows(rows, rows, {"agent7": "toolu_agent7"})
    assert len(expanded) == 2
    assert hook_module.get_tool_use_id_from_task_notification(expanded[1]) == "toolu_agent7"
    assert len(hook_module.expand_task_stop_rows(rows, rows, {})) == 1


def test_taskstop_in_a_later_firing_releases_the_deferred_turn(hook_module, fake_langfuse, tmp_path):
    rows = workflow_turn_rows({"message": "Successfully stopped task: wtask1", "task_id": "wtask1",
                               "task_type": "local_workflow"})
    transcript = tmp_path / "transcript.jsonl"
    stop_index = next(i for i, r in enumerate(rows) if r["uuid"] == "m-u2")
    write(transcript, rows[:stop_index] + [user("m-human", ts(5), "how is it going?"),
                                          asst("m-ah", ts(5, 1), "msg_mh", [{"type": "text", "text": "running"}], 1, "end_turn")])
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" not in roots_by_user_uuid(fake_langfuse)  # deferred, waiting
    write(transcript, rows[stop_index:], mode="a")
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" in roots_by_user_uuid(fake_langfuse)


def test_agent_settles_when_it_stops_its_own_workflow(hook_module, fake_langfuse, tmp_path):
    rows = [
        user("x-u", ts(0, 5), "go"),
        asst("x-a1", ts(1), "msg_x1", [{"type": "tool_use", "id": "toolu_wf", "name": "Workflow",
                                        "input": {"script": "r"}}], 3, "tool_use"),
        user("x-r1", ts(1, 1), [{"type": "tool_result", "tool_use_id": "toolu_wf", "content": "launched"}],
             toolUseResult={"status": "async_launched", "isAsync": True, "taskId": "wtask9", "runId": "wf_none"}),
        asst("x-a2", ts(1, 2), "msg_x2", [{"type": "tool_use", "id": "toolu_stop", "name": "TaskStop",
                                           "input": {"task_id": "wtask9"}}], 3, "tool_use"),
        user("x-r2", ts(1, 3), [{"type": "tool_result", "tool_use_id": "toolu_stop", "content": "stopped"}],
             toolUseResult={"message": "Successfully stopped task: wtask9", "task_id": "wtask9"}),
        asst("x-a3", ts(1, 4), "msg_x3", [{"type": "text", "text": "x done"}], 8, "end_turn"),
    ]
    transcript = setup_session(tmp_path, rows)
    fire(hook_module, fake_langfuse, transcript)
    assert len(spans_of(fake_langfuse, "Agent x")) == 1


def test_deferred_turn_queue_ages_out(hook_module, fake_langfuse, tmp_path):
    rows = workflow_turn_rows("Error: not running")
    transcript = tmp_path / "transcript.jsonl"
    stop_index = next(i for i, r in enumerate(rows) if r["uuid"] == "m-u2")
    write(transcript, rows[:stop_index] + [user("m-human", ts(5), "status?"),
                                          asst("m-ah", ts(5, 1), "msg_mh", [{"type": "text", "text": "running"}], 1, "end_turn")])
    fire(hook_module, fake_langfuse, transcript)  # turn 1 goes to the deferred queue
    assert "start a workflow" not in roots_by_user_uuid(fake_langfuse)
    write(transcript, [user("m-late", ts(0, 0, 23), "hours later"),
                       asst("m-al", ts(0, 1, 23), "msg_ml", [{"type": "text", "text": "ok"}], 1, "end_turn"),
                       user("m-late2", ts(1, 0, 23), "again")], mode="a")
    fire(hook_module, fake_langfuse, transcript)
    assert "start a workflow" in roots_by_user_uuid(fake_langfuse)
