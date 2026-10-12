from __future__ import annotations

import base64
import json
import os


PNG_B64 = base64.b64encode(b"\x89PNG" + b"x" * 3000).decode()


def image_block(data: str = PNG_B64, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


# ---- text extraction (direct message content) ----

def test_extract_text_replaces_image_blocks_with_markers(hook_module):
    text = hook_module.extract_text_from_content(
        [image_block(), {"type": "text", "text": "[Image #1] what is this?"}]
    )

    assert PNG_B64 not in text
    assert "[image image/png ~2KB]" in text
    assert "[Image #1] what is this?" in text


def test_extract_text_of_image_only_content_is_not_empty(hook_module):
    text = hook_module.extract_text_from_content([image_block()])

    assert text == "[image image/png ~2KB]"


def test_image_marker_survives_malformed_source(hook_module):
    assert hook_module.extract_text_from_content([{"type": "image"}]) == "[image unknown type]"
    assert (
        hook_module.extract_text_from_content([{"type": "image", "source": {"type": "base64"}}])
        == "[image unknown type]"
    )


# ---- tool_result serialization ----

def test_tool_result_images_become_markers_and_trailing_text_survives(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", False)
    big_b64 = "A" * (hook_module.MAX_CHARS * 2)
    result = hook_module.get_tool_result_for_observation(
        {
            "content": [
                {"type": "text", "text": "Screenshot captured"},
                image_block(data=big_b64),
                {"type": "text", "text": "tab context after the image"},
            ],
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )

    assert isinstance(result.output, str)
    assert big_b64[:100] not in result.output
    assert "Screenshot captured" in result.output
    assert "tab context after the image" in result.output
    assert result.output_meta == {"truncated": False, "orig_len": len(result.output)}


def test_tool_result_string_content_stays_plain(hook_module):
    result = hook_module.get_tool_result_for_observation(
        {"content": "plain text result", "timestamp": "2026-01-01T00:00:01.000Z"}
    )

    assert result.output == "plain text result"


def test_tool_result_capture_appends_media_after_text(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", True)
    result = hook_module.get_tool_result_for_observation(
        {
            "content": [
                {"type": "text", "text": "Screenshot captured"},
                image_block(),
            ],
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )

    text, media = result.output[0], result.output[1:]
    assert "Screenshot captured" in text
    assert PNG_B64 not in text
    assert [(m.content_type, m.content_bytes) for m in media] == [("image/png", base64.b64decode(PNG_B64))]


def test_tool_result_with_multiple_images_keeps_marker_and_media_order(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", True)
    b64_a = base64.b64encode(b"\x89PNG-first" * 300).decode()
    b64_b = base64.b64encode(b"RIFF-second" * 500).decode()
    result = hook_module.get_tool_result_for_observation(
        {
            "content": [
                {"type": "text", "text": "first shot"},
                image_block(data=b64_a),
                {"type": "text", "text": "second shot"},
                image_block(data=b64_b, media_type="image/webp"),
            ],
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )

    text, media = result.output[0], result.output[1:]
    assert (
        text.index("first shot")
        < text.index("[image image/png")
        < text.index("second shot")
        < text.index("[image image/webp")
    )
    assert [(m.content_type, m.content_bytes) for m in media] == [
        ("image/png", base64.b64decode(b64_a)),
        ("image/webp", base64.b64decode(b64_b)),
    ]


def test_tool_result_final_content_images_become_markers(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", False)
    result = hook_module.get_tool_result_for_observation(
        {
            "content": "Async agent launched successfully.",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "final_content": [image_block(), {"type": "text", "text": "final answer"}],
            "final_timestamp": "2026-01-01T00:00:05.000Z",
        }
    )

    assert PNG_B64 not in json.dumps(result.final_output)
    assert "final answer" in result.final_output


# ---- media creation guardrails ----

def test_media_creation_is_on_by_default(hook_module):
    assert hook_module.CAPTURE_IMAGES is True
    media = hook_module.media_from_image_block(image_block())
    assert media is not None and media.content_type == "image/png"


def test_media_creation_rejects_non_base64_and_survives_sdk_errors(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", True)

    url_block = {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
    assert hook_module.media_from_image_block(url_block) is None

    # The SDK does not raise on these, so the hook must reject them itself.
    assert hook_module.media_from_image_block(image_block(data="!!!not-base64!!!")) is None
    assert hook_module.media_from_image_block(image_block(media_type="")) is None

    # The SDK accepts whitespace-wrapped base64, so the hook must too.
    wrapped = "\n".join(PNG_B64[i:i + 76] for i in range(0, len(PNG_B64), 76))
    media = hook_module.media_from_image_block(image_block(data=wrapped))
    assert media is not None and media.content_bytes == base64.b64decode(PNG_B64)

    class ExplodingMedia:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(hook_module, "LangfuseMedia", ExplodingMedia)
    assert hook_module.media_from_image_block(image_block()) is None

    monkeypatch.setattr(hook_module, "LangfuseMedia", None)
    assert hook_module.media_from_image_block(image_block()) is None


# ---- SDK media auto-upload hardening ----

def _config(hook_module):
    return hook_module.LangfuseConfig(public_key="pk", secret_key="sk", host="http://x", user_id=None)


def test_capture_off_disables_sdk_media_auto_upload(hook_module, monkeypatch):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", False)
    monkeypatch.delenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", raising=False)

    try:
        hook_module.create_langfuse_client(_config(hook_module))
        assert os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] == "false"
    finally:
        os.environ.pop("LANGFUSE_MEDIA_UPLOAD_ENABLED", None)


def test_capture_on_leaves_sdk_media_upload_untouched(hook_module, monkeypatch):
    monkeypatch.delenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", raising=False)
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", True)

    hook_module.create_langfuse_client(_config(hook_module))

    assert "LANGFUSE_MEDIA_UPLOAD_ENABLED" not in os.environ


def test_user_set_media_upload_setting_wins_over_hardening(hook_module, monkeypatch):
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")

    hook_module.create_langfuse_client(_config(hook_module))

    assert os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] == "true"


# ---- turn root input ----

def test_turn_root_input_attaches_media_when_capture_is_on(hook_module, fake_langfuse, monkeypatch, tmp_path):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", True)
    turn = hook_module.build_turns(
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:01:00.000Z",
                "uuid": "user-1",
                "message": {
                    "role": "user",
                    "content": [image_block(), {"type": "text", "text": "what is this?"}],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:01:01.000Z",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "a png"}]},
            },
        ]
    )[0]

    hook_module.open_turn_root_span(fake_langfuse, "session-img", 1, turn, tmp_path / "t.jsonl")

    root = fake_langfuse.observations[-1]
    text, media = root.kwargs["input"]["content"][0], root.kwargs["input"]["content"][1:]
    assert "what is this?" in text
    assert PNG_B64 not in text
    assert [(m.content_type, m.content_bytes) for m in media] == [("image/png", base64.b64decode(PNG_B64))]


def test_turn_root_input_stays_text_when_capture_is_off(hook_module, fake_langfuse, monkeypatch, tmp_path):
    monkeypatch.setattr(hook_module, "CAPTURE_IMAGES", False)
    turn = hook_module.build_turns(
        [
            {
                "type": "user",
                "timestamp": "2026-01-01T00:01:00.000Z",
                "uuid": "user-1",
                "message": {
                    "role": "user",
                    "content": [image_block(), {"type": "text", "text": "what is this?"}],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:01:01.000Z",
                "uuid": "assistant-1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "a png"}]},
            },
        ]
    )[0]

    hook_module.open_turn_root_span(fake_langfuse, "session-img", 1, turn, tmp_path / "t.jsonl")

    root = fake_langfuse.observations[-1]
    content = root.kwargs["input"]["content"]
    assert isinstance(content, str)
    assert "[image image/png ~2KB]" in content
    assert "what is this?" in content
