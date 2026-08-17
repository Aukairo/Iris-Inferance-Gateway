import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.inference.tool_handler import (
    format_tool_system_prompt,
    parse_tool_calls,
    extract_reasoning_content,
)
from app.inference.engine import model_wrapper
from app.batching.scheduler import RequestContext


def test_extract_reasoning_content():
    raw = "<think>Let me calculate 2+2. It equals 4.</think>The answer is 4."
    reasoning, clean = extract_reasoning_content(raw)
    assert reasoning == "Let me calculate 2+2. It equals 4."
    assert clean == "The answer is 4."

    no_think = "Hello, world!"
    reasoning2, clean2 = extract_reasoning_content(no_think)
    assert reasoning2 is None
    assert clean2 == "Hello, world!"


def test_parse_tool_calls_mistral():
    raw_text = 'I will call the search tool. [TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "MLX Framework"}}] [/TOOL_CALLS]'
    clean_content, tool_calls, finish_reason = parse_tool_calls(raw_text)

    assert finish_reason == "tool_calls"
    assert clean_content == "I will call the search tool."
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "web_search"
    assert json.loads(tool_calls[0].function.arguments) == {"query": "MLX Framework"}


def test_parse_tool_calls_llama():
    raw_text = '<call:get_weather>{"location": "Tokyo"}</call:get_weather>'
    clean_content, tool_calls, finish_reason = parse_tool_calls(raw_text)

    assert finish_reason == "tool_calls"
    assert clean_content is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"location": "Tokyo"}


def test_format_tool_system_prompt_tool_choice():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate math expression",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }
        }
    ]

    # tool_choice = none
    prompt_none = format_tool_system_prompt(tools, tool_choice="none")
    assert prompt_none == ""

    # tool_choice = required
    prompt_req = format_tool_system_prompt(tools, tool_choice="required")
    assert "IMPORTANT: You MUST call at least one of the tools" in prompt_req

    # tool_choice = specific function
    prompt_specific = format_tool_system_prompt(
        tools,
        tool_choice={"type": "function", "function": {"name": "calculator"}}
    )
    assert "IMPORTANT: You MUST call the tool 'calculator'" in prompt_specific


@pytest.fixture
def mock_engine_setup():
    import os
    from app.core.key_manager import KeyManager

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.encode = MagicMock(return_value=[10, 20, 30])
    mock_tokenizer.eos_token_ids = [0]

    test_keys_file = "data/test_compat_keys.json"
    if os.path.exists(test_keys_file):
        try:
            os.remove(test_keys_file)
        except Exception:
            pass
    test_manager = KeyManager(filepath=test_keys_file)
    test_manager.init_store()

    with patch.object(model_wrapper, "load_model"), \
         patch.object(model_wrapper, "unload_model"), \
         patch.object(model_wrapper, "model", mock_model), \
         patch.object(model_wrapper, "tokenizer", mock_tokenizer), \
         patch.object(model_wrapper, "model_name", "deepseek-r1"), \
         patch("app.api.auth.key_manager", test_manager), \
         patch("app.main.key_manager", test_manager):

        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer
        }

    if os.path.exists(test_keys_file):
        try:
            os.remove(test_keys_file)
        except Exception:
            pass


def test_reasoning_model_chat_completion(mock_engine_setup):
    app = create_app()

    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 8
        response_text = "<think>Analysing user request...</think>Paris is the capital of France."
        request_ctx.response_queue.put_nowait((response_text, "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "model": "deepseek-r1",
                "messages": [{"role": "user", "content": "What is the capital of France?"}]
            }
            res = client.post("/v1/chat/completions", json=payload)
            assert res.status_code == 200
            data = res.json()
            message = data["choices"][0]["message"]
            assert message["content"] == "Paris is the capital of France."
            assert message["reasoning_content"] == "Analysing user request..."


def test_text_completions_endpoint(mock_engine_setup):
    app = create_app()

    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 4
        request_ctx.response_queue.put_nowait(("World!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "model": "text-davinci-003",
                "prompt": "Hello "
            }
            res = client.post("/v1/completions", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["object"] == "text_completion"
            assert data["choices"][0]["text"] == "World!"
