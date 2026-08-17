import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.inference.tool_handler import format_tool_system_prompt, parse_tool_calls
from app.inference.engine import model_wrapper
from app.batching.scheduler import RequestContext


def test_format_tool_system_prompt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    prompt = format_tool_system_prompt(tools)
    assert "<tools>" in prompt
    assert "get_weather" in prompt
    assert "<tool_call>" in prompt


def test_parse_tool_calls_qwen_xml():
    raw_text = 'Sure! Here is the weather: <tool_call>\n{"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}\n</tool_call>'
    clean_content, tool_calls, finish_reason = parse_tool_calls(raw_text)
    
    assert finish_reason == "tool_calls"
    assert clean_content == "Sure! Here is the weather:"
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"location": "San Francisco, CA"}


def test_parse_tool_calls_json_block():
    raw_text = '```json\n{"name": "calculator", "arguments": {"expression": "42 * 2"}}\n```'
    clean_content, tool_calls, finish_reason = parse_tool_calls(raw_text)
    
    assert finish_reason == "tool_calls"
    assert clean_content is None
    assert tool_calls is not None
    assert tool_calls[0].function.name == "calculator"
    assert json.loads(tool_calls[0].function.arguments) == {"expression": "42 * 2"}


def test_parse_tool_calls_no_tool():
    raw_text = "The capital of France is Paris."
    clean_content, tool_calls, finish_reason = parse_tool_calls(raw_text)
    
    assert finish_reason == "stop"
    assert clean_content == "The capital of France is Paris."
    assert tool_calls is None


@pytest.fixture
def mock_engine_for_tools():
    import os
    from app.core.key_manager import KeyManager

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value=[10, 20, 30])
    mock_tokenizer.eos_token_ids = [0]
    
    test_keys_file = "data/test_tools_keys.json"
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
         patch.object(model_wrapper, "model_name", "qwen-2.5-7b"), \
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


def test_api_tool_calls_non_streaming(mock_engine_for_tools):
    app = create_app()

    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 5
        response_text = '<tool_call>\n{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}\n</tool_call>'
        request_ctx.response_queue.put_nowait((response_text, "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What is the stock price of AAPL?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_stock_price",
                            "description": "Fetch current stock price",
                            "parameters": {
                                "type": "object",
                                "properties": {"symbol": {"type": "string"}},
                                "required": ["symbol"]
                            }
                        }
                    }
                ]
            }
            res = client.post("/v1/chat/completions", json=payload)
            assert res.status_code == 200
            data = res.json()
            assert data["model"] == "gpt-4o"
            choice = data["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_stock_price"
            assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"symbol": "AAPL"}


def test_api_dynamic_model_endpoint(mock_engine_for_tools):
    app = create_app()
    with TestClient(app) as client:
        res = client.get("/v1/models/claude-3-5-sonnet")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "claude-3-5-sonnet"
        assert data["object"] == "model"
