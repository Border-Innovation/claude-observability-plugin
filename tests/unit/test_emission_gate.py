from __future__ import annotations

NOTIFICATION = (
    "<task-notification>\n<task-id>t1</task-id>\n"
    "<tool-use-id>toolu_X</tool-use-id>\n<result>done</result>\n</task-notification>"
)


def test_queued_notification_counts_as_undelivered(hook_module):
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == ["toolu_X"]


def test_delivery_clears_the_queued_notification(hook_module):
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
        {"type": "user", "origin": {"kind": "task-notification"},
         "message": {"role": "user", "content": NOTIFICATION}},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == []


def test_rows_without_notifications_pass_the_gate(hook_module):
    rows = [
        {"type": "user", "message": {"role": "user", "content": "hallo"}},
        {"type": "queue-operation", "operation": "remove", "content": None},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == []


def test_remove_with_content_clears_a_notification_routed_elsewhere(hook_module):
    # A subagent's child notifies: enqueued on the main queue, removed at once
    # (routed to that subagent), never delivered as a main user row.
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
        {"type": "queue-operation", "operation": "remove", "content": NOTIFICATION},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == []


def test_remove_clears_one_enqueue_only(hook_module):
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
        {"type": "queue-operation", "operation": "remove", "content": NOTIFICATION},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == ["toolu_X"]


def test_request_echo_of_a_notification_is_not_a_delivery(hook_module):
    rows = [
        {"type": "queue-operation", "operation": "enqueue", "content": NOTIFICATION},
        {"type": "api-request-blob", "message": {"role": "user", "content": NOTIFICATION}},
    ]
    assert hook_module.get_undelivered_queued_notification_ids(rows) == ["toolu_X"]
