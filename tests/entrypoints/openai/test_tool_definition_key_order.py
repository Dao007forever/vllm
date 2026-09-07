# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunctionDefinition must dump its keys in the order the client sent them.

Chat templates such as GLM's iterate the dumped tool dict when rendering
the tool list into the prompt, so pydantic's declaration order would
otherwise silently rewrite model-visible prompt text.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def _request_with_function(function: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": function}],
    )


@pytest.mark.parametrize(
    "keys",
    [
        ("name", "description", "parameters"),
        ("description", "name", "parameters"),
        ("parameters", "name", "description"),
    ],
)
def test_function_definition_dump_preserves_client_key_order(keys):
    values = {
        "name": "collect_preferences",
        "description": "Ask the user questions.",
        "parameters": {"type": "object", "properties": {}},
    }
    request = _request_with_function({k: values[k] for k in keys})

    dumped = request.tools[0].model_dump()["function"]

    assert list(dumped) == list(keys)
    assert dumped == values


def test_function_definition_dump_drops_unset_optional_fields():
    request = _request_with_function({"description": "d", "name": "f"})

    dumped = request.tools[0].model_dump()["function"]

    assert list(dumped) == ["description", "name", "parameters"]
    assert "strict" not in dumped
