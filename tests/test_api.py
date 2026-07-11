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
