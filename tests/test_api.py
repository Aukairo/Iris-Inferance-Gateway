import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import create_app
from app.inference.engine import model_wrapper
from app.batching.scheduler import RequestContext


@pytest.fixture
def mock_engine_and_scheduler():
    """
    Fixture to mock the MLX model engine and scheduler startup behavior to isolate API routing.
    """
    import os
    from app.core.key_manager import KeyManager

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
    mock_tokenizer.eos_token_ids = [0]
    
    mock_detok = MagicMock()
    mock_detok.last_segment = "token text"
    mock_tokenizer.detokenizer = mock_detok

    test_keys_file = "data/test_api_keys.json"
    if os.path.exists(test_keys_file):
        try:
            os.remove(test_keys_file)
        except Exception:
            pass
    test_manager = KeyManager(filepath=test_keys_file)
    test_manager.init_store()

    with patch.object(model_wrapper, "load_model") as mock_load, \
         patch.object(model_wrapper, "unload_model") as mock_unload, \
         patch.object(model_wrapper, "model", mock_model), \
         patch.object(model_wrapper, "tokenizer", mock_tokenizer), \
         patch.object(model_wrapper, "model_name", "mock-qwen-model"), \
         patch("app.api.auth.key_manager", test_manager), \
         patch("app.main.key_manager", test_manager):
        
        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "load": mock_load,
            "unload": mock_unload
        }

    if os.path.exists(test_keys_file):
        try:
            os.remove(test_keys_file)
        except Exception:
            pass


def test_health_endpoint(mock_engine_and_scheduler) -> None:
    """
    Verify that the health check endpoint returns correct stats.
    """
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "mock-qwen-model"
        assert "uptime_seconds" in data
        assert "memory_usage_mb" in data


def test_metrics_endpoint(mock_engine_and_scheduler) -> None:
    """
    Verify that the Prometheus metrics endpoint responds with the correct content type.
    """
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


def test_chat_completions_non_streaming(mock_engine_and_scheduler) -> None:
    """
    Test a non-streaming chat completion request.
    Mocks the scheduler queue addition to feed response tokens immediately.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        # Simulate background worker feeding tokens into response queue
        request_ctx.completion_tokens = 2
        request_ctx.response_queue.put_nowait(("Hello ", None))
        request_ctx.response_queue.put_nowait(("World!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [
                    {"role": "user", "content": "Tell me a joke."}
                ],
                "temperature": 0.5,
                "max_tokens": 100,
                "stream": False
            }
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["content"] == "Hello World!"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["usage"]["prompt_tokens"] == 3
            assert data["usage"]["completion_tokens"] == 2
            assert data["usage"]["total_tokens"] == 5


def test_chat_completions_streaming(mock_engine_and_scheduler) -> None:
    """
    Test a streaming chat completion request.
    Verifies that SSE format chunks are correctly generated and streamed.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        # Simulate streaming chunks
        request_ctx.response_queue.put_nowait(("Hello ", None))
        request_ctx.response_queue.put_nowait(("World!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": True
            }
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            # Read and verify the streaming response lines
            lines = response.text.split("\n")
            chunks = [line for line in lines if line.startswith("data: ")]
            
            # Should have the initial role, two token segments, final chunk, and DONE
            assert len(chunks) >= 4
            assert "role" in chunks[0]
            assert "Hello" in chunks[1]
            assert "World!" in chunks[2]
            assert "[DONE]" in lines[-3] or "[DONE]" in lines[-2]


def test_models_endpoints(mock_engine_and_scheduler) -> None:
    """
    Verify /v1/models and /models list endpoints.
    """
    app = create_app()
    with TestClient(app) as client:
        # Test with /v1/models
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "mock-qwen-model"
        
        # Test with /models
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "mock-qwen-model"


def test_get_model_details(mock_engine_and_scheduler) -> None:
    """
    Verify retrieving specific model details by ID.
    """
    app = create_app()
    with TestClient(app) as client:
        # Success case
        response = client.get("/v1/models/mock-qwen-model")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mock-qwen-model"
        assert data["object"] == "model"
        
        # Error case
        response = client.get("/v1/models/non-existent-model")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


def test_path_normalization_middleware(mock_engine_and_scheduler) -> None:
    """
    Verify that PathNormalizationMiddleware collapses consecutive slashes.
    """
    app = create_app()
    with TestClient(app) as client:
        # //models should resolve to /models
        response = client.get("http://testserver//models")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
        
        # /v1//models should resolve to /v1/models
        response = client.get("/v1//models")
        assert response.status_code == 200
        assert response.json()["object"] == "list"
        
        # ///health should resolve to /health
        response = client.get("///health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # /v1/chat/completions/models should resolve to /v1/models
        response = client.get("/v1/chat/completions/models")
        assert response.status_code == 200
        assert response.json()["object"] == "list"


def test_self_healing_chat_completions(mock_engine_and_scheduler) -> None:
    """
    Verify that the misconfigured base URL path for chat completions is self-healed.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 1
        request_ctx.response_queue.put_nowait(("Self-healed works!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            # Post to /v1/chat/completions/chat/completions
            response = client.post("/v1/chat/completions/chat/completions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Self-healed works!"


def test_chat_completions_alias(mock_engine_and_scheduler) -> None:
    """
    Verify chat completions work via the alias /chat/completions.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 1
        request_ctx.response_queue.put_nowait(("Hi", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            response = client.post("/chat/completions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "Hi"


def test_schema_robustness(mock_engine_and_scheduler) -> None:
    """
    Verify that ChatCompletionRequest handles unrecognized extra fields
    and null content values without throwing a 422 validation error.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 1
        request_ctx.response_queue.put_nowait(("Schema works!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]}
                ],
                "stream": False,
                "extra_unsupported_key": "some_value",
                "tools": [{"type": "function"}],
                "tool_choice": "auto"
            }
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Schema works!"


def test_content_blocks_support(mock_engine_and_scheduler) -> None:
    """
    Verify that ChatCompletionRequest handles content blocks (list of text dicts)
    successfully and converts them to string content.
    """
    app = create_app()
    
    def mock_add_request(request_ctx: RequestContext) -> None:
        request_ctx.completion_tokens = 1
        request_ctx.response_queue.put_nowait(("Blocks work!", "stop"))

    with TestClient(app) as client:
        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            payload = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Hello, "},
                        {"type": "text", "text": "world!"}
                    ]}
                ],
                "stream": False
            }
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Blocks work!"


def test_normalize_messages_util() -> None:
    """
    Directly test routes.normalize_messages to ensure it:
    1. Alternates user/assistant roles.
    2. Converts tool messages to user.
    3. Merges consecutive same-role messages.
    """
    from app.api.routes import normalize_messages
    from app.models.schemas import ChatMessage
    
    messages = [
        ChatMessage(role="system", content="System instruction"),
        ChatMessage(role="user", content="User message 1"),
        ChatMessage(role="assistant", content=None, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "test_func", "arguments": "{}"}}]),
        ChatMessage(role="tool", content="Tool result", name="test_func"),
        ChatMessage(role="user", content="User message 2")
    ]
    
    normalized = normalize_messages(messages)
    assert len(normalized) == 4
    assert normalized[0] == {"role": "system", "content": "System instruction"}
    assert normalized[1] == {"role": "user", "content": "User message 1"}
    assert normalized[2] == {"role": "assistant", "content": "[Call Tool: test_func({})]"}
    assert normalized[3] == {"role": "user", "content": "[Tool Response for 'test_func']:\nTool result\n\nUser message 2"}
