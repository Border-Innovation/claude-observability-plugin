"""Claude Code 2.1.301+ writes echo rows of every API request (api-request,
api-request-shape, api-request-blob). A blob carries message.role but no id,
usage or timestamp; it must never be read as a chat row."""
from __future__ import annotations

from typing import Any


def blob(role: str, content: Any) -> dict:
    return {"type": "api-request-blob", "sessionId": "s", "hash": "0" * 64,
            "message": {"role": role, "content": content}}


def test_blob_is_not_a_chat_row(hook_module):
    for role in ("user", "assistant", "system"):
        assert hook_module.get_user_or_assistant_role_from_row(blob(role, "x")) is None


def test_untyped_rows_keep_the_message_role_fallback(hook_module):
    assert hook_module.get_user_or_assistant_role_from_row({"message": {"role": "assistant"}}) == "assistant"


def test_echo_rows_are_recognised(hook_module):
    for row_type in ("api-request", "api-request-blob", "api-request-shape"):
        assert hook_module.is_request_echo_row({"type": row_type})
    assert not hook_module.is_request_echo_row({"type": "assistant"})
    assert not hook_module.is_request_echo_row({"message": {"role": "user"}})


def test_blob_echoing_a_notification_is_not_a_notification(hook_module):
    text = "<task-notification><task-id>a1</task-id><result>r</result></task-notification>"
    assert not hook_module.is_task_notification_row(blob("user", text))
    assert hook_module.is_task_notification_row({"type": "user", "message": {"role": "user", "content": text}})


def test_blobs_neither_split_turns_nor_add_generations(hook_module):
    rows = [
        {"type": "user", "uuid": "u1", "timestamp": "2026-08-01T10:00:00.000Z",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "timestamp": "2026-08-01T10:00:01.000Z",
         "message": {"id": "m1", "role": "assistant", "content": [{"type": "text", "text": "one"}],
                     "usage": {"input_tokens": 1, "output_tokens": 2}}},
        blob("user", "go"),
        blob("assistant", [{"type": "text", "text": "one"}]),
        {"type": "assistant", "uuid": "a2", "timestamp": "2026-08-01T10:00:02.000Z",
         "message": {"id": "m2", "role": "assistant", "content": [{"type": "text", "text": "two"}],
                     "usage": {"input_tokens": 1, "output_tokens": 3}}},
    ]
    turns = hook_module.build_turns(rows)
    assert len(turns) == 1
    assert [hook_module.get_message_id(m) for m in turns[0].assistant_msgs] == ["m1", "m2"]
    assert all(hook_module.get_usage_details_from_row(m) for m in turns[0].assistant_msgs)


def test_control_with_the_old_role_fallback_blobs_split_the_turn(hook_module):
    # What the unfixed reader did: a nested message.role made any row a chat row.
    rows = [
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": "one"}},
        {"message": {"role": "user", "content": "go"}},
        {"message": {"role": "assistant", "content": "one"}},
    ]
    turns = hook_module.build_turns(rows)
    assert len(turns) == 2
    assert hook_module.get_message_id(turns[1].assistant_msgs[0]) is None


def test_raw_unicode_line_separator_in_a_string_keeps_the_row(hook_module, tmp_path):
    # JSON permits U+2028 / U+0085 unescaped inside strings; str.splitlines()
    # splits on them and the row with the final usage was dropped.
    import json as _json
    rows = [
        {"type": "assistant", "message": {"id": "m1", "role": "assistant",
                                          "content": [{"type": "text", "text": "a b\x85c"}],
                                          "usage": {"output_tokens": 7}}},
    ]
    path = tmp_path / "agent.jsonl"
    path.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    assert " " in path.read_text(encoding="utf-8")
    read = hook_module.read_subagent_jsonl(path)
    assert [r["message"]["usage"]["output_tokens"] for r in read] == [7]


def test_control_splitlines_drops_that_row(tmp_path):
    import json as _json
    text = _json.dumps({"t": "a b"}, ensure_ascii=False)
    parsed = []
    for line in text.splitlines():
        try:
            parsed.append(_json.loads(line))
        except ValueError:
            pass
    assert parsed == []
