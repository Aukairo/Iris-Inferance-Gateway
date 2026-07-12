import os
import json
import uuid
import threading
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from app.config.config import settings
from app.core.logging import logger
from app.core.config_manager import config_manager

class ApiKeyRecord(BaseModel):
    key: str
    name: str
    token_cap: int = Field(-1, description="Max total tokens allowed. -1 means unlimited.")
    amount_cap: float = Field(-1.0, description="Max total cost allowed. -1.0 means unlimited.")
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    cached_tokens_used: int = 0
    total_tokens_used: int = 0
    amount_spent: float = 0.0
    active: bool = True
    revoked: bool = False


class KeyManager:
    """
    Manages API keys, token caps, and usage logs.
    Synchronizes state thread-safely with a local JSON file store.
    """
    def __init__(self, filepath: str = "data/keys.json") -> None:
        self.filepath = filepath
        self.lock = threading.Lock()
        self.keys: Dict[str, ApiKeyRecord] = {}

    def init_store(self) -> None:
        """
        Creates storage directory and file, and pre-loads keys.
        """
        with self.lock:
            # Create data folder if it doesn't exist
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for k, v in data.items():
                            self.keys[k] = ApiKeyRecord(**v)
                    logger.info("Loaded %d API keys from storage.", len(self.keys))
                except Exception as e:
                    logger.error("Failed to load API keys file: %s. Re-initializing...", str(e))
                    self.keys = {}
            else:
                self.keys = {}
                self._save_unlocked()

            # For backward compatibility: if a static api_key is configured in settings
            # and doesn't exist in the JSON database yet, add it automatically
            if settings.api_key and settings.api_key not in self.keys:
                default_key = settings.api_key
                self.keys[default_key] = ApiKeyRecord(
                    key=default_key,
                    name="Default Config Key",
                    token_cap=-1,
                    active=True
                )
                self._save_unlocked()
                logger.info("Registered default static API key in storage.")

    async def init_store_async(self) -> None:
        from app.core.mongodb import mongodb_manager
        if mongodb_manager.enabled and mongodb_manager.db is not None:
            try:
                cursor = mongodb_manager.db["api_keys"].find({})
                keys_data = {}
                async for doc in cursor:
                    key_val = doc.get("key")
                    if key_val:
                        record_dict = {k: v for k, v in doc.items() if k != "_id"}
                        keys_data[key_val] = ApiKeyRecord(**record_dict)
                
                with self.lock:
                    self.keys = keys_data
                logger.info("Loaded %d API keys from MongoDB.", len(self.keys))
                
                # Check for default static key as well
                if settings.api_key and settings.api_key not in self.keys:
                    default_key = settings.api_key
                    record = ApiKeyRecord(
                        key=default_key,
                        name="Default Config Key",
                        token_cap=-1,
                        active=True
                    )
                    with self.lock:
                        self.keys[default_key] = record
                    await mongodb_manager.db["api_keys"].replace_one(
                        {"key": default_key},
                        record.model_dump(),
                        upsert=True
                    )
                    logger.info("Registered default static API key in MongoDB.")
                return
            except Exception as e:
                logger.error("Failed to load API keys from MongoDB: %s. Falling back to file store.", str(e))
        
        # Fallback to sync file loading
        self.init_store()

    def _save_unlocked(self) -> None:
        """
        Internal save helper (called while holding the lock).
        """
        from app.core.mongodb import mongodb_manager
        if mongodb_manager.enabled:
            return
            
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(
                    {k: v.model_dump() for k, v in self.keys.items()},
                    f,
                    indent=4,
                    ensure_ascii=False
                )
        except Exception as e:
            logger.error("Failed to save API keys to file %s: %s", self.filepath, str(e))

    def verify_key(self, key: str) -> Optional[ApiKeyRecord]:
        """
        Validates the key and checks if its limit is not exceeded.
        """
        with self.lock:
            record = self.keys.get(key)
            if record and record.active and not getattr(record, "revoked", False):
                # Check limits
                if record.token_cap != -1 and record.total_tokens_used >= record.token_cap:
                    logger.warning("API Key %s token limit exceeded (%d/%d)", record.name, record.total_tokens_used, record.token_cap)
                if record.amount_cap != -1.0 and record.amount_spent >= record.amount_cap:
                    logger.warning("API Key %s budget limit exceeded ($%.4f/$%.4f)", record.name, record.amount_spent, record.amount_cap)
                return record
            return None

    def record_usage(self, key: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> None:
        """
        Updates prompt/completion/cached token usage counters and amount spent.
        """
        with self.lock:
            record = self.keys.get(key)
            if record:
                record.prompt_tokens_used += prompt_tokens
                record.completion_tokens_used += completion_tokens
                
                # Check if fields exist/initialized (backwards compatibility)
                if not hasattr(record, "cached_tokens_used") or record.cached_tokens_used is None:
                    record.cached_tokens_used = 0
                record.cached_tokens_used += cached_tokens
                
                record.total_tokens_used += (prompt_tokens + completion_tokens)
                
                # Calculate cost based on dynamic configuration
                pricing = config_manager.get_pricing()
                price_input = pricing.price_per_1m_input_tokens
                price_output = pricing.price_per_1m_output_tokens
                price_cached = pricing.price_per_1m_cached_tokens
                
                uncached_prompt = max(0, prompt_tokens - cached_tokens)
                cost = (
                    (uncached_prompt * price_input) +
                    (cached_tokens * price_cached) +
                    (completion_tokens * price_output)
                ) / 1_000_000.0
                
                if not hasattr(record, "amount_spent") or record.amount_spent is None:
                    record.amount_spent = 0.0
                record.amount_spent += cost
                
                self._save_unlocked()
                logger.info(
                    "Recorded token usage for key %s. Total: %d, Cap: %d, Cost: $%.6f, Budget Cap: $%.4f",
                    record.name,
                    record.total_tokens_used,
                    record.token_cap,
                    record.amount_spent,
                    record.amount_cap
                )
                
                # Update in MongoDB if enabled
                from app.core.mongodb import mongodb_manager
                if mongodb_manager.enabled:
                    rec_dict = record.model_dump()
                    async def _db_update():
                        try:
                            await mongodb_manager.db["api_keys"].replace_one(
                                {"key": rec_dict["key"]},
                                rec_dict,
                                upsert=True
                            )
                        except Exception as e:
                            logger.error("Failed to update API key usage in MongoDB: %s", str(e))
                    mongodb_manager.run_async(_db_update())

    def create_key(self, name: str, token_cap: int = -1, amount_cap: float = -1.0) -> ApiKeyRecord:
        """
        Generates a new API key and stores it.
        """
        with self.lock:
            # Generate a secure key prefix sk-
            generated_key = f"sk-{uuid.uuid4().hex}"
            record = ApiKeyRecord(
                key=generated_key,
                name=name,
                token_cap=token_cap,
                amount_cap=amount_cap,
                active=True
            )
            self.keys[generated_key] = record
            self._save_unlocked()
            logger.info("Created new API key for user: %s with token_cap=%d, amount_cap=%.4f", name, token_cap, amount_cap)
            
            from app.core.mongodb import mongodb_manager
            if mongodb_manager.enabled:
                rec_dict = record.model_dump()
                async def _db_update():
                    try:
                        await mongodb_manager.db["api_keys"].replace_one(
                            {"key": rec_dict["key"]},
                            rec_dict,
                            upsert=True
                        )
                    except Exception as e:
                        logger.error("Failed to save new key to MongoDB: %s", str(e))
                mongodb_manager.run_async(_db_update())
                
            return record

    def delete_key(self, key: str) -> bool:
        """
        Soft-deletes/revokes the API key to preserve usage history.
        """
        with self.lock:
            if key in self.keys:
                record = self.keys[key]
                record.active = False
                record.revoked = True
                self._save_unlocked()
                logger.info("Revoked/Soft-deleted API key: %s (%s)", key, record.name)
                
                from app.core.mongodb import mongodb_manager
                if mongodb_manager.enabled:
                    rec_dict = record.model_dump()
                    async def _db_update():
                        try:
                            await mongodb_manager.db["api_keys"].replace_one(
                                {"key": rec_dict["key"]},
                                rec_dict,
                                upsert=True
                            )
                        except Exception as e:
                            logger.error("Failed to update revoked status in MongoDB: %s", str(e))
                    mongodb_manager.run_async(_db_update())
                return True
            return False

    def update_key_cap(self, key: str, token_cap: int, amount_cap: float = -1.0) -> bool:
        """
        Updates the token and budget caps.
        """
        with self.lock:
            record = self.keys.get(key)
            if record:
                record.token_cap = token_cap
                record.amount_cap = amount_cap
                self._save_unlocked()
                logger.info("Updated caps for key %s to token_cap=%d, amount_cap=%.4f", record.name, token_cap, amount_cap)
                
                from app.core.mongodb import mongodb_manager
                if mongodb_manager.enabled:
                    rec_dict = record.model_dump()
                    async def _db_update():
                        try:
                            await mongodb_manager.db["api_keys"].replace_one(
                                {"key": rec_dict["key"]},
                                rec_dict,
                                upsert=True
                            )
                        except Exception as e:
                            logger.error("Failed to update key caps in MongoDB: %s", str(e))
                    mongodb_manager.run_async(_db_update())
                return True
            return False

    def rotate_key(self, old_key: str) -> Optional[str]:
        """
        Generates a new bearer token string for a client, keeping usage stats intact.
        """
        import uuid
        with self.lock:
            if old_key in self.keys:
                record = self.keys.pop(old_key)
                new_key = f"sk-{uuid.uuid4().hex}"
                record.key = new_key
                self.keys[new_key] = record
                self._save_unlocked()
                logger.info("Rotated API key for %s. Old key: %s, New key: %s", record.name, old_key, new_key)
                
                from app.core.mongodb import mongodb_manager
                if mongodb_manager.enabled:
                    rec_dict = record.model_dump()
                    async def _db_update():
                        try:
                            await mongodb_manager.db["api_keys"].delete_one({"key": old_key})
                            await mongodb_manager.db["api_keys"].replace_one(
                                {"key": rec_dict["key"]},
                                rec_dict,
                                upsert=True
                            )
                        except Exception as e:
                            logger.error("Failed to rotate key in MongoDB: %s", str(e))
                    mongodb_manager.run_async(_db_update())
                return new_key
            return None

    def reset_key_usage(self, key: str) -> bool:
        """
        Resets token counters and amount spent back to zero.
        """
        with self.lock:
            record = self.keys.get(key)
            if record:
                record.prompt_tokens_used = 0
                record.completion_tokens_used = 0
                record.cached_tokens_used = 0
                record.total_tokens_used = 0
                record.amount_spent = 0.0
                self._save_unlocked()
                logger.info("Reset token usage and spending counters for key %s", record.name)
                
                from app.core.mongodb import mongodb_manager
                if mongodb_manager.enabled:
                    rec_dict = record.model_dump()
                    async def _db_update():
                        try:
                            await mongodb_manager.db["api_keys"].replace_one(
                                {"key": rec_dict["key"]},
                                rec_dict,
                                upsert=True
                            )
                        except Exception as e:
                            logger.error("Failed to reset key usage in MongoDB: %s", str(e))
                    mongodb_manager.run_async(_db_update())
                return True
            return False

    def list_keys(self) -> List[ApiKeyRecord]:
        """
        Returns a list of all key records.
        """
        with self.lock:
            return list(self.keys.values())


# Global key manager instance
key_manager = KeyManager()
