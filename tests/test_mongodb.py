import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from app.core.mongodb import MongoDBManager

@pytest.mark.asyncio
async def test_mongodb_manager_disabled_by_default():
    manager = MongoDBManager()
    assert manager.enabled is False
    assert manager.client is None
    assert manager.db is None
    
    # Save operations should return early and not raise errors when disabled
    await manager.save_request_log(
        request_id="test_req",
        api_key=None,
        model="test_model",
        prompt=[{"role": "user", "content": "hello"}],
        response="world",
        prompt_tokens=5,
        completion_tokens=5,
        cached_tokens=0,
        latency_ms=100,
        stream=False,
        status="success"
    )
    
    logs = await manager.get_request_logs()
    assert logs == {"total": 0, "logs": []}
    
    stats = await manager.get_stats()
    assert stats == {"total_requests": 0, "success_rate": 1.0, "total_cost": 0.0}

@pytest.mark.asyncio
@patch("app.core.mongodb.AsyncIOMotorClient")
async def test_mongodb_manager_init(mock_client):
    manager = MongoDBManager()
    manager.init_db("mongodb://mock:27017", "test_db")
    
    assert manager.enabled is True
    assert manager.client is not None
    assert manager.db is not None
    mock_client.assert_called_once_with("mongodb://mock:27017", serverSelectionTimeoutMS=3000)

@pytest.mark.asyncio
async def test_mongodb_save_and_retrieve_log():
    # Setup mocks
    mock_db = MagicMock()
    mock_collection = AsyncMock()
    mock_db.__getitem__.return_value = mock_collection
    
    manager = MongoDBManager()
    manager.client = MagicMock()
    manager.db = mock_db
    manager.enabled = True
    
    # Mock counter increment
    mock_collection.find_one_and_update.return_value = {"value": 42}
    
    # 1. Test counter increment
    req_num = await manager.get_next_request_number()
    assert req_num == 42
    
    # 2. Test saving log
    with patch("app.core.mongodb.key_manager") as mock_key_mgr:
        # Mock API Key record lookup
        mock_key_record = MagicMock()
        mock_key_record.name = "Test User"
        mock_key_mgr.keys.get.return_value = mock_key_record
        
        await manager.save_request_log(
            request_id="req_123",
            api_key="sk-test",
            model="mlx-model",
            prompt=[{"role": "user", "content": "hi"}],
            response="hello",
            prompt_tokens=10,
            completion_tokens=20,
            cached_tokens=5,
            latency_ms=150,
            stream=True,
            status="success"
        )
        
        # Verify document inserted
        mock_collection.insert_one.assert_called_once()
        inserted_doc = mock_collection.insert_one.call_args[0][0]
        
        assert inserted_doc["request_id"] == "req_123"
        assert inserted_doc["request_number"] == 42
        assert inserted_doc["client_name"] == "Test User"
        assert inserted_doc["api_key"] == "sk-test"
        assert inserted_doc["model"] == "mlx-model"
        assert inserted_doc["prompt"] == [{"role": "user", "content": "hi"}]
        assert inserted_doc["response"] == "hello"
        assert inserted_doc["prompt_tokens"] == 10
        assert inserted_doc["completion_tokens"] == 20
        assert inserted_doc["cached_tokens"] == 5
        assert inserted_doc["total_tokens"] == 30
        assert inserted_doc["latency_ms"] == 150
        assert inserted_doc["stream"] is True
        assert inserted_doc["status"] == "success"
        assert isinstance(inserted_doc["timestamp"], datetime)
