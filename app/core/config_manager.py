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

    async def init_store_async(self) -> None:
        from app.core.mongodb import mongodb_manager
        if mongodb_manager.enabled and mongodb_manager.db is not None:
            try:
                doc = await mongodb_manager.db["config"].find_one({"_id": "pricing"})
                if doc:
                    data = {k: v for k, v in doc.items() if k != "_id"}
                    with self.lock:
                        self.pricing = ModelPricingConfig(**data)
                    logger.info("Loaded model pricing config from MongoDB: %s", self.pricing)
                    return
                else:
                    # Initialize default values in MongoDB
                    with self.lock:
                        self.pricing = ModelPricingConfig(
                            price_per_1m_input_tokens=settings.price_per_1m_input_tokens,
                            price_per_1m_output_tokens=settings.price_per_1m_output_tokens,
                            price_per_1m_cached_tokens=settings.price_per_1m_cached_tokens
                        )
                    await mongodb_manager.db["config"].replace_one(
                        {"_id": "pricing"},
                        {"_id": "pricing", **self.pricing.model_dump()},
                        upsert=True
                    )
                    logger.info("Initialized default pricing config in MongoDB.")
                    return
            except Exception as e:
                logger.error("Failed to load config from MongoDB: %s. Falling back to file store.", str(e))
        
        # Fallback to sync file loading
        self.init_store()

    def _load_defaults(self) -> None:
        self.pricing = ModelPricingConfig(
            price_per_1m_input_tokens=settings.price_per_1m_input_tokens,
            price_per_1m_output_tokens=settings.price_per_1m_output_tokens,
            price_per_1m_cached_tokens=settings.price_per_1m_cached_tokens
        )
        self._save_unlocked()
        logger.info("Initialized default pricing config.")

    def _save_unlocked(self) -> None:
        from app.core.mongodb import mongodb_manager
        if mongodb_manager.enabled:
            return
            
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
            
            # If MongoDB is enabled, update it in the database
            from app.core.mongodb import mongodb_manager
            if mongodb_manager.enabled:
                rec_dict = self.pricing.model_dump()
                async def _db_update():
                    try:
                        await mongodb_manager.db["config"].replace_one(
                            {"_id": "pricing"},
                            {"_id": "pricing", **rec_dict},
                            upsert=True
                        )
                        logger.info("Persisted pricing config to MongoDB.")
                    except Exception as e:
                        logger.error("Failed to persist pricing config to MongoDB: %s", str(e))
                mongodb_manager.run_async(_db_update())
                
            return self.pricing


# Global config manager instance
config_manager = ConfigManager()
