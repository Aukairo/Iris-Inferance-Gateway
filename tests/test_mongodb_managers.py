import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import json
from app.core.config_manager import ConfigManager, ModelPricingConfig
from app.core.key_manager import KeyManager, ApiKeyRecord

TEST_KEYS_FILE = "data/test_mongo_keys.json"
TEST_CONFIG_FILE = "data/test_mongo_config.json"

@pytest.fixture(autouse=True)
def cleanup_test_files():
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)
    if os.path.exists(TEST_CONFIG_FILE):
        os.remove(TEST_CONFIG_FILE)
    yield
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)
    if os.path.exists(TEST_CONFIG_FILE):
        os.remove(TEST_CONFIG_FILE)


@pytest.fixture
def mock_mongodb():
    with patch("app.core.mongodb.mongodb_manager") as mock_mgr:
        mock_mgr.enabled = True
        mock_db = MagicMock()
        mock_mgr.db = mock_db
        # Mock collections
        mock_config_col = AsyncMock()
        mock_keys_col = MagicMock()
        mock_keys_col.insert_one = AsyncMock()
        mock_keys_col.replace_one = AsyncMock()
        mock_keys_col.delete_one = AsyncMock()
        
        def get_collection(name):
            if name == "config":
                return mock_config_col
            elif name == "api_keys":
                return mock_keys_col
            return MagicMock()
            
        mock_db.__getitem__.side_effect = get_collection
        import asyncio
        mock_mgr.run_async.side_effect = lambda coro: asyncio.create_task(coro)
        yield mock_mgr, mock_config_col, mock_keys_col


@pytest.mark.asyncio
async def test_config_manager_mongodb(mock_mongodb):
    mock_mgr, mock_config_col, mock_keys_col = mock_mongodb
    
    # 1. Test init_store_async when config exists in DB
    mock_config_col.find_one.return_value = {
        "_id": "pricing",
        "price_per_1m_input_tokens": 12.0,
        "price_per_1m_output_tokens": 34.0,
        "price_per_1m_cached_tokens": 5.0
    }
    
    config_mgr = ConfigManager(filepath=TEST_CONFIG_FILE)
    await config_mgr.init_store_async()
    
    assert config_mgr.get_pricing().price_per_1m_input_tokens == 12.0
    assert config_mgr.get_pricing().price_per_1m_output_tokens == 34.0
    assert config_mgr.get_pricing().price_per_1m_cached_tokens == 5.0
    
    # Verify no local file was created
    assert not os.path.exists(TEST_CONFIG_FILE)
    
    # 2. Test update_pricing propagates to DB and doesn't write local file
    config_mgr.update_pricing(50.0, 60.0, 20.0)
    assert config_mgr.get_pricing().price_per_1m_input_tokens == 50.0
    assert not os.path.exists(TEST_CONFIG_FILE)
    
    # Verify run_async was called to execute the update
    mock_mongodb[0].run_async.assert_called_once()


@pytest.mark.asyncio
async def test_key_manager_mongodb(mock_mongodb):
    mock_mgr, mock_config_col, mock_keys_col = mock_mongodb
    
    # Mock find cursor for keys
    mock_cursor = MagicMock()
    # Mock async generator
    async def mock_async_gen():
        yield {
            "key": "sk-mongodb-test-key-1",
            "name": "MongoDB Developer",
            "token_cap": 5000,
            "amount_cap": 2.5,
            "prompt_tokens_used": 100,
            "completion_tokens_used": 200,
            "cached_tokens_used": 10,
            "total_tokens_used": 300,
            "amount_spent": 0.05,
            "active": True,
            "revoked": False
        }
    mock_cursor.__aiter__.side_effect = mock_async_gen
    mock_keys_col.find.return_value = mock_cursor
    
    key_mgr = KeyManager(filepath=TEST_KEYS_FILE)
    await key_mgr.init_store_async()
    
    # Verify key loaded correctly from MongoDB
    record = key_mgr.verify_key("sk-mongodb-test-key-1")
    assert record is not None
    assert record.name == "MongoDB Developer"
    assert record.token_cap == 5000
    assert not os.path.exists(TEST_KEYS_FILE)
    
    # Test create_key
    new_rec = key_mgr.create_key("New DB Key", token_cap=1000)
    assert new_rec.name == "New DB Key"
    assert new_rec.key in key_mgr.keys
    assert not os.path.exists(TEST_KEYS_FILE)
    
    # Test delete_key
    key_mgr.delete_key("sk-mongodb-test-key-1")
    assert key_mgr.keys["sk-mongodb-test-key-1"].revoked is True
    assert not os.path.exists(TEST_KEYS_FILE)
