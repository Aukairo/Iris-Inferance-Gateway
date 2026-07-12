import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
from app.config.config import settings
from app.core.logging import logger
from app.core.key_manager import key_manager
from app.core.config_manager import config_manager

class MongoDBManager:
    """
    Manages MongoDB connections and performs database logging/retrieval operations.
    Designed to fail gracefully: if MongoDB is unavailable, endpoints will still run.
    """
    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.enabled = False
        self.loop = None

    def init_db(self, uri: str, db_name: str) -> None:
        """
        Initializes the MongoDB client.
        Does not raise exceptions if connection fails, keeping the server operational.
        """
        try:
            logger.info("Connecting to MongoDB at %s...", uri)
            self.client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=3000)
            self.db = self.client[db_name]
            self.enabled = True
            try:
                import asyncio
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None
            logger.info("MongoDB client initialized for database '%s'.", db_name)
        except Exception as e:
            logger.error("Failed to initialize MongoDB: %s. Database logging will be disabled.", str(e))
            self.client = None
            self.db = None
            self.enabled = False
            self.loop = None

    def run_async(self, coro) -> None:
        """
        Executes a coroutine asynchronously.
        If a running event loop is captured (self.loop), schedules it thread-safely.
        Otherwise, schedules it on the current thread's event loop if possible,
        or spawns a temporary event loop.
        """
        if not self.enabled or self.db is None:
            return
        
        try:
            import asyncio
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self.loop)
            else:
                try:
                    current_loop = asyncio.get_running_loop()
                    if current_loop.is_running():
                        current_loop.create_task(coro)
                        return
                except RuntimeError:
                    pass
                
                # Standalone thread/test case with no running loop
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
        except Exception as e:
            logger.error("Failed to run async DB operation: %s", str(e))

    async def close_db(self) -> None:
        """
        Closes the active MongoDB connection.
        """
        if self.client:
            logger.info("Closing MongoDB connection...")
            self.client.close()
            logger.info("MongoDB connection closed.")
            self.enabled = False

    async def get_next_request_number(self) -> int:
        """
        Atomically increments and retrieves the next sequential request number using
        a dedicated counters collection. Returns 0 if MongoDB is disabled.
        """
        if not self.enabled or self.db is None:
            return 0
        try:
            counter = await self.db["counters"].find_one_and_update(
                {"_id": "request_number"},
                {"$inc": {"value": 1}},
                upsert=True,
                return_document=pymongo.ReturnDocument.AFTER
            )
            return counter["value"]
        except Exception as e:
            logger.error("Failed to increment request counter in MongoDB: %s", str(e))
            return 0

    async def save_request_log(
        self,
        request_id: str,
        api_key: Optional[str],
        model: str,
        prompt: Optional[List[Dict[str, Any]]],
        response: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        latency_ms: int,
        stream: bool,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Persists detailed API request/response log along with metadata into MongoDB.
        Calculates real-time USD billing cost based on active settings.
        """
        if not self.enabled or self.db is None:
            return

        try:
            # Resolve Client Name/Label
            client_name = "Anonymous"
            cost = 0.0
            if api_key:
                record = key_manager.keys.get(api_key)
                if record:
                    client_name = record.name
                else:
                    client_name = "Config Key" if api_key == settings.api_key else "Unknown Key"
                
                # Fetch current model pricing rates from config_manager
                pricing = config_manager.get_pricing()
                price_input = pricing.price_per_1m_input_tokens
                price_output = pricing.price_per_1m_output_tokens
                price_cached = pricing.price_per_1m_cached_tokens
                
                # Deduct cached tokens from input token count to count uncached ones
                uncached_prompt = max(0, prompt_tokens - cached_tokens)
                cost = (
                    (uncached_prompt * price_input) +
                    (cached_tokens * price_cached) +
                    (completion_tokens * price_output)
                ) / 1_000_000.0

            # Fetch the sequential request number
            request_number = await self.get_next_request_number()

            log_doc = {
                "request_number": request_number,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc),
                "api_key": api_key,
                "client_name": client_name,
                "model": model,
                "prompt": prompt,
                "response": response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost,
                "latency_ms": latency_ms,
                "stream": stream,
                "status": status,
                "error_message": error_message
            }

            await self.db["request_logs"].insert_one(log_doc)
            logger.info("Successfully logged request #%d (%s) to MongoDB.", request_number, request_id)
        except Exception as e:
            logger.error("Failed to save request log to MongoDB: %s", str(e))

    async def get_request_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
        client_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves a paginated list of logged requests, sorted in descending order of request number.
        """
        if not self.enabled or self.db is None:
            return {"total": 0, "logs": []}

        try:
            query = {}
            if status_filter:
                query["status"] = status_filter
            if client_filter:
                query["client_name"] = client_filter

            cursor = self.db["request_logs"].find(query)
            cursor = cursor.sort("request_number", pymongo.DESCENDING).skip(offset).limit(limit)
            
            logs = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                if isinstance(doc.get("timestamp"), datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat() + "Z"
                logs.append(doc)

            total = await self.db["request_logs"].count_documents(query)
            return {"total": total, "logs": logs}
        except Exception as e:
            logger.error("Failed to fetch request logs from MongoDB: %s", str(e))
            return {"total": 0, "logs": []}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Aggregates statistical indicators across all request records.
        """
        if not self.enabled or self.db is None:
            return {"total_requests": 0, "success_rate": 1.0, "total_cost": 0.0}

        try:
            total_requests = await self.db["request_logs"].count_documents({})
            successful_requests = await self.db["request_logs"].count_documents({"status": "success"})
            
            # Aggregate cost spent
            pipeline = [{"$group": {"_id": None, "total_cost": {"$sum": "$cost"}}}]
            agg = await self.db["request_logs"].aggregate(pipeline).to_list(length=1)
            total_cost = agg[0]["total_cost"] if agg else 0.0
            
            success_rate = (successful_requests / total_requests) if total_requests > 0 else 1.0
            return {
                "total_requests": total_requests,
                "success_rate": round(success_rate, 4),
                "total_cost": round(total_cost, 6)
            }
        except Exception as e:
            logger.error("Failed to fetch statistics from MongoDB: %s", str(e))
            return {"total_requests": 0, "success_rate": 1.0, "total_cost": 0.0}


# Global MongoDB manager instance
mongodb_manager = MongoDBManager()
