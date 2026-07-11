import os
import json
import threading
from typing import Optional
from pydantic import BaseModel
from app.config.config import settings
from app.core.logging import logger

class ModelPricingConfig(BaseModel):
    price_per_1m_input_tokens: float
    price_per_1m_output_tokens: float
    price_per_1m_cached_tokens: float


class ConfigManager:
    """
    Manages global dynamic server configurations, such as model pricing.
    Persists data thread-safely in a local JSON file.
    """
    def __init__(self, filepath: str = "data/config.json") -> None:
        self.filepath = filepath
        self.lock = threading.Lock()
        self.pricing: Optional[ModelPricingConfig] = None
        
    def init_store(self) -> None:
        with self.lock:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.pricing = ModelPricingConfig(**data)
                    logger.info("Loaded model pricing config from storage: %s", self.pricing)
                except Exception as e:
                    logger.error("Failed to load config file: %s. Re-initializing...", str(e))
                    self._load_defaults()
            else:
                self._load_defaults()

    def _load_defaults(self) -> None:
        self.pricing = ModelPricingConfig(
            price_per_1m_input_tokens=settings.price_per_1m_input_tokens,
            price_per_1m_output_tokens=settings.price_per_1m_output_tokens,
            price_per_1m_cached_tokens=settings.price_per_1m_cached_tokens
        )
        self._save_unlocked()
        logger.info("Initialized default pricing config.")

    def _save_unlocked(self) -> None:
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.pricing.model_dump(), f, indent=4)
        except Exception as e:
            logger.error("Failed to save config to file %s: %s", self.filepath, str(e))

    def get_pricing(self) -> ModelPricingConfig:
        with self.lock:
            if self.pricing is None:
                # If not initialized, try loading or fallback
                try:
                    if os.path.exists(self.filepath):
                        with open(self.filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self.pricing = ModelPricingConfig(**data)
                            return self.pricing
                except Exception:
                    pass
                self.pricing = ModelPricingConfig(
                    price_per_1m_input_tokens=settings.price_per_1m_input_tokens,
                    price_per_1m_output_tokens=settings.price_per_1m_output_tokens,
                    price_per_1m_cached_tokens=settings.price_per_1m_cached_tokens
                )
            return self.pricing

    def update_pricing(self, price_input: float, price_output: float, price_cached: float) -> ModelPricingConfig:
        with self.lock:
            self.pricing = ModelPricingConfig(
                price_per_1m_input_tokens=price_input,
                price_per_1m_output_tokens=price_output,
                price_per_1m_cached_tokens=price_cached
            )
            self._save_unlocked()
            logger.info("Updated dynamic model pricing: %s", self.pricing)
            return self.pricing


# Global config manager instance
config_manager = ConfigManager()
