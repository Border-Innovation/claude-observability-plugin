from __future__ import annotations

import json


def test_span_id_depends_on_trace_and_key_only(hook_module):
    a = hook_module.derive_span_id("ab" * 16, "gen:msg_1")
    assert a == hook_module.derive_span_id("ab" * 16, "gen:msg_1")
    assert a != hook_module.derive_span_id("ab" * 16, "gen:msg_2")
    assert a != hook_module.derive_span_id("cd" * 16, "gen:msg_1")
    assert 0 < a < 2 ** 64


def test_positional_keys_are_scoped(hook_module):
    with hook_module.id_scope("agent1:0"):
        first = hook_module.derive_span_id("ab" * 16, "gen:noid:0")
    with hook_module.id_scope("agent1:1"):
        second = hook_module.derive_span_id("ab" * 16, "gen:noid:0")
    assert first != second
    # Natural ids ignore the scope, so a turn re-indexing cannot move them.
    with hook_module.id_scope("agent1:0"):
        natural = hook_module.derive_span_id("ab" * 16, "gen:msg_1")
    assert natural == hook_module.derive_span_id("ab" * 16, "gen:msg_1")


def test_preset_id_is_one_shot(hook_module, fake_langfuse):
    generator = hook_module.install_deterministic_span_ids(fake_langfuse)
    assert generator is hook_module.install_deterministic_span_ids(fake_langfuse)
    generator.next_span_id = 0xABC
    assert generator.generate_span_id() == 0xABC
    assert generator.generate_span_id() != 0xABC


def test_same_key_same_parent_same_observation_id(hook_module, fake_langfuse):
    parent = hook_module.carrier_span("12" * 16, "34" * 8)
    ids = [
        hook_module._start_backdated(fake_langfuse, name="x", as_type="span", start_time=None,
                                     parent_otel_span=parent, id_key="agent:a1").id
        for _ in range(2)
    ]
    assert ids[0] == ids[1] == f"{hook_module.derive_span_id('12' * 16, 'agent:a1'):016x}"


def test_control_without_a_key_ids_stay_random(hook_module, fake_langfuse):
    parent = hook_module.carrier_span("12" * 16, "34" * 8)
    ids = {
        hook_module._start_backdated(fake_langfuse, name="x", as_type="span", start_time=None,
                                     parent_otel_span=parent).id
        for _ in range(2)
    }
    assert len(ids) == 2


def test_corrupt_registry_starts_empty(hook_module, tmp_path):
    path = tmp_path / "langfuse_agents.0123456789abcdef.json"
    path.write_text("{not json", encoding="utf-8")
    registry = hook_module.AgentRegistry.load(path)
    assert registry.entries == {}
    registry.entries["a"] = {"size": 1}
    registry.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"agents": {"a": {"size": 1}}}
