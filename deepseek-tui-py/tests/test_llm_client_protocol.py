"""Protocol-level tests for the DeepSeek chat-completions client."""

import deepseek_tui.llm.client as llm_client

from deepseek_tui.llm import ContentBlock, Message, MessageRequest


def _request(messages, **kwargs):
    return MessageRequest(
        model=kwargs.pop("model", "deepseek-v4-pro"),
        messages=messages,
        max_tokens=4096,
        reasoning_effort=kwargs.pop("reasoning_effort", "high"),
        **kwargs,
    )


class TestToolNameEncoding:
    def test_tool_name_roundtrip_with_dot(self):
        original = "multi_tool_use.parallel"
        encoded = llm_client._to_api_tool_name(original)
        assert encoded == "multi_tool_use-x00002E-parallel"
        assert llm_client._from_api_tool_name(encoded) == original

    def test_tool_name_decode_mangled_hex_escape(self):
        mangled = "multi_tool_use.x00002E-parallel"
        assert llm_client._from_api_tool_name(mangled) == "multi_tool_use..parallel"


class TestMessageSerialization:
    def test_serialization_replays_reasoning_and_tool_result(self):
        request = _request(
            [
                Message(role="user", content=[ContentBlock(type="text", text="list files")]),
                Message(
                    role="assistant",
                    content=[
                        ContentBlock(type="thinking", thinking="I should call list_dir."),
                        ContentBlock(
                            type="tool_use",
                            id="call_1",
                            name="list_dir",
                            input={"path": "."},
                        ),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ContentBlock(
                            type="tool_result",
                            tool_use_id="call_1",
                            content="README.md",
                        )
                    ],
                ),
            ]
        )

        messages = llm_client.DeepSeekClient._serialize_messages(request)

        assert messages[0] == {"role": "user", "content": "list files"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["reasoning_content"] == "I should call list_dir."
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "README.md",
        }

    def test_serialization_omits_reasoning_when_thinking_disabled(self):
        request = _request(
            [
                Message(
                    role="assistant",
                    content=[
                        ContentBlock(type="thinking", thinking="internal"),
                        ContentBlock(type="text", text="done"),
                    ],
                )
            ],
            reasoning_effort="off",
        )

        messages = llm_client.DeepSeekClient._serialize_messages(request)

        assert messages == [{"role": "assistant", "content": "done"}]

    def test_serialization_strips_orphaned_tool_calls_without_results(self):
        request = _request(
            [
                Message(
                    role="assistant",
                    content=[
                        ContentBlock(type="thinking", thinking="Need a tool."),
                        ContentBlock(
                            type="tool_use",
                            id="call_orphan",
                            name="read_file",
                            input={"path": "README.md"},
                        ),
                    ],
                )
            ]
        )

        messages = llm_client.DeepSeekClient._serialize_messages(request)

        assert messages == []


class TestResponseParsing:
    def test_parse_response_keeps_reasoning_and_decodes_tool_name(self):
        response = llm_client.DeepSeekClient._parse_response(
            {
                "id": "chatcmpl-1",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "reasoning_content": "Need to inspect the file.",
                            "content": "Working on it.",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "function": {
                                        "name": "multi_tool_use-x00002E-parallel",
                                        "arguments": '{"q":"hello"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        assert [block.type for block in response.content] == [
            "thinking",
            "text",
            "tool_use",
        ]
        assert response.content[0].thinking == "Need to inspect the file."
        assert response.content[2].name == "multi_tool_use.parallel"
        assert response.content[2].input == {"q": "hello"}
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5


class TestSseChunkParsing:
    def test_parse_sse_chunk_streams_thinking_then_tool_use(self):
        content_index = [0]
        text_started = [False]
        thinking_started = [False]
        tool_indices = {}

        thinking_events = llm_client.DeepSeekClient._parse_sse_chunk(
            {
                "choices": [
                    {"delta": {"reasoning_content": "I should call read_file."}}
                ]
            },
            content_index,
            text_started,
            thinking_started,
            tool_indices,
            True,
        )

        assert thinking_events[0]["event"].type == "content_block_start"
        assert thinking_events[0]["event"].content_block.type == "thinking"
        assert thinking_events[1]["event"].delta.thinking == "I should call read_file."

        tool_events = llm_client.DeepSeekClient._parse_sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            content_index,
            text_started,
            thinking_started,
            tool_indices,
            True,
        )

        assert tool_events[0]["event"].type == "content_block_stop"
        assert tool_events[1]["event"].type == "content_block_start"
        assert tool_events[1]["event"].content_block.type == "tool_use"
        assert tool_events[1]["event"].content_block.id == "call_x"
        assert tool_events[2]["event"].delta.partial_json == '{"path":"README.md"}'

        finish_events = llm_client.DeepSeekClient._parse_sse_chunk(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7},
            },
            content_index,
            text_started,
            thinking_started,
            tool_indices,
            True,
        )

        assert finish_events[0]["event"].type == "content_block_stop"
        assert finish_events[1]["event"].type == "message_delta"
        assert finish_events[1]["event"].message_delta.stop_reason == "tool_calls"
        assert finish_events[1]["event"].usage.output_tokens == 7

    def test_parallel_tool_calls_get_unique_fallback_ids(self):
        events = llm_client.DeepSeekClient._parse_sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "list_dir",
                                        "arguments": '{"path":"a"}',
                                    },
                                },
                                {
                                    "index": 1,
                                    "function": {
                                        "name": "list_dir",
                                        "arguments": '{"path":"b"}',
                                    },
                                },
                            ]
                        }
                    }
                ]
            },
            [0],
            [False],
            [False],
            {},
            False,
        )

        starts = [
            item["event"].content_block.id
            for item in events
            if item["event"].type == "content_block_start"
        ]
        assert len(starts) == 2
        assert starts[0] != starts[1]
