import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Model Configuration
    model_name: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    # Server Network Options
    host: str = "127.0.0.1"
    port: int = 8000

    # Default Inference Parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048

    # Batching Scheduler Configuration
    batching_window_ms: int = 25
    max_batch_size: int = 16

    # Security Configuration
    api_key: Optional[str] = None
    admin_password: str = "admin123"

    # Logging and Observability
    logging_level: str = "INFO"
    metrics_enabled: bool = True

    # Pricing Configuration (per 1,000,000 tokens)
    price_per_1m_input_tokens: float = 0.15
    price_per_1m_output_tokens: float = 0.60
    price_per_1m_cached_tokens: float = 0.075

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()
