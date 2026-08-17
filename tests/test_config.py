import os
import pytest
from app.config.config import Settings


def test_default_config() -> None:
    """
    Verify that default settings match configuration specifications.
    """
    settings = Settings()
    assert isinstance(settings.model_name, str) and len(settings.model_name) > 0
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.temperature == 0.7
    assert settings.top_p == 0.9
    assert settings.max_tokens == 2048
    assert settings.batching_window_ms == 25
    assert settings.max_batch_size == 16
    assert settings.api_key is None or settings.api_key == ""
    assert settings.logging_level == "INFO"
    assert settings.metrics_enabled is True


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that environment variables correctly override configuration defaults.
    """
    monkeypatch.setenv("MODEL_NAME", "mlx-community/gemma-3-4b-it-4bit")
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("TEMPERATURE", "0.2")
    monkeypatch.setenv("TOP_P", "0.95")
    monkeypatch.setenv("MAX_TOKENS", "512")
    monkeypatch.setenv("BATCHING_WINDOW_MS", "50")
    monkeypatch.setenv("MAX_BATCH_SIZE", "32")
    monkeypatch.setenv("API_KEY", "test-secret-key")
    monkeypatch.setenv("LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("METRICS_ENABLED", "false")

    settings = Settings()
    assert settings.model_name == "mlx-community/gemma-3-4b-it-4bit"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.temperature == 0.2
    assert settings.top_p == 0.95
    assert settings.max_tokens == 512
    assert settings.batching_window_ms == 50
    assert settings.max_batch_size == 32
    assert settings.api_key == "test-secret-key"
    assert settings.logging_level == "DEBUG"
    assert settings.metrics_enabled is False
