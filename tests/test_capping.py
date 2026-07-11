import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import create_app
from app.config.config import settings
from app.core.key_manager import KeyManager, ApiKeyRecord
from app.core.config_manager import ConfigManager
from app.api.auth import verify_api_key
from app.inference.engine import model_wrapper

TEST_KEYS_FILE = "data/test_keys.json"
TEST_CONFIG_FILE = "data/test_config.json"

@pytest.fixture(autouse=True)
def setup_test_key_manager():
    """
    Clean up test keys file before and after tests.
    """
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)
        
    test_manager = KeyManager(filepath=TEST_KEYS_FILE)
    test_manager.init_store()
    
    # Patch the global key_manager with our test instance
    with patch("app.api.auth.key_manager", test_manager), \
         patch("app.api.admin_routes.key_manager", test_manager), \
         patch("app.main.key_manager", test_manager), \
         patch("app.batching.scheduler.key_manager", test_manager):
        yield test_manager
        
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)


@pytest.fixture(autouse=True)
def setup_test_config_manager():
    """
    Clean up test config file before and after tests.
    """
    if os.path.exists(TEST_CONFIG_FILE):
        os.remove(TEST_CONFIG_FILE)
        
    test_config = ConfigManager(filepath=TEST_CONFIG_FILE)
    test_config.init_store()
    
    with patch("app.core.key_manager.config_manager", test_config), \
         patch("app.api.admin_routes.config_manager", test_config), \
         patch("app.main.config_manager", test_config):
        yield test_config
        
    if os.path.exists(TEST_CONFIG_FILE):
        os.remove(TEST_CONFIG_FILE)


def test_key_manager_crud(setup_test_key_manager, setup_test_config_manager):
    mgr = setup_test_key_manager
    cfg = setup_test_config_manager
    
    # Create key with token cap and amount cap
    record = mgr.create_key(name="Developer A", token_cap=1000, amount_cap=5.0)
    assert record.name == "Developer A"
    assert record.token_cap == 1000
    assert record.amount_cap == 5.0
    assert record.total_tokens_used == 0
    assert record.amount_spent == 0.0
    assert record.cached_tokens_used == 0
    assert record.active is True
    
    # List keys
    keys = mgr.list_keys()
    assert len(keys) == 1
    assert keys[0].key == record.key
    
    # Verify key
    verified = mgr.verify_key(record.key)
    assert verified is not None
    assert verified.name == "Developer A"
    
    # Record usage with default cache hit rates
    mgr.record_usage(record.key, prompt_tokens=100, completion_tokens=200, cached_tokens=40)
    verified = mgr.verify_key(record.key)
    # Default Cost = (60 * 0.15 + 40 * 0.075 + 200 * 0.60) / 1,000,000 = 0.000132
    assert abs(verified.amount_spent - 0.000132) < 1e-9
    
    # Update config model pricing dynamically
    cfg.update_pricing(price_input=1.0, price_output=2.0, price_cached=0.5)
    
    # Rotate the key
    old_key = record.key
    new_key = mgr.rotate_key(old_key)
    assert new_key is not None
    assert new_key != old_key
    
    # Verify old key no longer works
    assert mgr.verify_key(old_key) is None

    # The new key should still verify and have preserved statistics
    verified = mgr.verify_key(new_key)
    assert verified.prompt_tokens_used == 100
    assert verified.completion_tokens_used == 200
    assert verified.cached_tokens_used == 40

    # Record more usage under new rates using the new key
    mgr.record_usage(new_key, prompt_tokens=100, completion_tokens=200, cached_tokens=40)
    verified = mgr.verify_key(new_key)
    # New cumulative cost = 0.000132 (first usage) + (60 * 1.0 + 40 * 0.5 + 200 * 2.0) / 1M = 0.000132 + 0.00048 = 0.000612
    assert abs(verified.amount_spent - 0.000612) < 1e-9

    # Reset usage stats
    mgr.reset_key_usage(new_key)
    verified = mgr.verify_key(new_key)
    assert verified.total_tokens_used == 0
    assert verified.amount_spent == 0.0

    # Soft-delete / revoke key
    mgr.delete_key(new_key)
    # The key is marked as inactive and revoked, but remains in the list
    assert len(mgr.list_keys()) == 1
    assert mgr.list_keys()[0].revoked is True
    assert mgr.list_keys()[0].active is False
    assert mgr.verify_key(new_key) is None  # Blocked from authentication


def test_admin_api_authorization(setup_test_key_manager):
    app = create_app()
    with TestClient(app) as client:
        # Request without header
        response = client.get("/api/admin/keys")
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid Admin Password."
        
        # Request with incorrect password
        response = client.get("/api/admin/keys", headers={"X-Admin-Password": "wrong-password"})
        assert response.status_code == 401
        
        # Request with correct password (default "admin123")
        response = client.get("/api/admin/keys", headers={"X-Admin-Password": "admin123"})
        assert response.status_code == 200
        assert response.json() == []


def test_admin_api_crud(setup_test_key_manager):
    app = create_app()
    with TestClient(app) as client:
        headers = {"X-Admin-Password": "admin123"}
        
        # Create Key with token_cap and amount_cap
        response = client.post(
            "/api/admin/keys",
            headers=headers,
            json={"name": "Production Client", "token_cap": 50, "amount_cap": 2.5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Production Client"
        assert data["token_cap"] == 50
        assert data["amount_cap"] == 2.5
        api_key = data["key"]
        
        # List Keys
        response = client.get("/api/admin/keys", headers=headers)
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["key"] == api_key
        
        # Update Cap
        response = client.put(
            f"/api/admin/keys/{api_key}/cap",
            headers=headers,
            json={"token_cap": 100, "amount_cap": 10.0}
        )
        assert response.status_code == 200
        
        # Verify updated cap
        response = client.get("/api/admin/keys", headers=headers)
        assert response.json()[0]["token_cap"] == 100
        assert response.json()[0]["amount_cap"] == 10.0

        # Rotate Key via Admin API
        response = client.post(f"/api/admin/keys/{api_key}/rotate", headers=headers)
        assert response.status_code == 200
        new_key = response.json()["new_key"]
        assert new_key != api_key

        # Verify list contains new key and stats/caps are preserved
        response = client.get("/api/admin/keys", headers=headers)
        assert len(response.json()) == 1
        assert response.json()[0]["key"] == new_key
        assert response.json()[0]["token_cap"] == 100
        assert response.json()[0]["amount_cap"] == 10.0

        # Reset Key stats via Admin API
        response = client.post(f"/api/admin/keys/{new_key}/reset", headers=headers)
        assert response.status_code == 200
        response = client.get("/api/admin/keys", headers=headers)
        assert response.json()[0]["total_tokens_used"] == 0

        # Revoke Rotated Key via Admin API
        response = client.delete(f"/api/admin/keys/{new_key}", headers=headers)
        assert response.status_code == 200

        # Verify key is still in list but marked as revoked
        response = client.get("/api/admin/keys", headers=headers)
        assert len(response.json()) == 1
        assert response.json()[0]["revoked"] is True


def test_admin_api_pricing_config(setup_test_config_manager):
    app = create_app()
    with TestClient(app) as client:
        headers = {"X-Admin-Password": "admin123"}
        
        # 1. Fetch default pricing config
        response = client.get("/api/admin/config/pricing", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["price_per_1m_input_tokens"] == settings.price_per_1m_input_tokens
        assert data["price_per_1m_output_tokens"] == settings.price_per_1m_output_tokens
        assert data["price_per_1m_cached_tokens"] == settings.price_per_1m_cached_tokens
        
        # 2. Update pricing config
        response = client.put(
            "/api/admin/config/pricing",
            headers=headers,
            json={
                "price_per_1m_input_tokens": 0.5,
                "price_per_1m_output_tokens": 1.5,
                "price_per_1m_cached_tokens": 0.25
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["price_per_1m_input_tokens"] == 0.5
        assert data["price_per_1m_output_tokens"] == 1.5
        assert data["price_per_1m_cached_tokens"] == 0.25
        
        # 3. Verify it persists on subsequent GET
        response = client.get("/api/admin/config/pricing", headers=headers)
        assert response.status_code == 200
        assert response.json()["price_per_1m_input_tokens"] == 0.5


def test_chat_completion_capping_limit(setup_test_key_manager):
    app = create_app()
    mgr = setup_test_key_manager
    
    # Create a key with a small cap of 50 tokens
    record = mgr.create_key(name="Small Budget User", token_cap=50, amount_cap=-1.0)
    api_key = record.key
    
    # Mock model templates
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
    mock_tokenizer.eos_token_ids = [0]
    
    with TestClient(app) as client, \
         patch.object(model_wrapper, "model", MagicMock()), \
         patch.object(model_wrapper, "tokenizer", mock_tokenizer):
        
        # Call with correct header - should authenticate successfully
        def mock_add_request(request_ctx):
            request_ctx.completion_tokens = 5
            request_ctx.response_queue.put_nowait(("Hello ", None))
            request_ctx.response_queue.put_nowait(("World!", "stop"))

        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            # Request under the cap
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 10,
                    "stream": False
                }
            )
            assert response.status_code == 200
            
        # Simulate usage exceeding the cap
        mgr.record_usage(api_key, prompt_tokens=30, completion_tokens=25) # 55 tokens > 50 cap
        
        # Request after exceeding cap - should receive 403 Forbidden
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello again!"}],
                "max_tokens": 10,
                "stream": False
            }
        )
        assert response.status_code == 403
        assert "tokens" in response.json()["detail"].lower()


def test_chat_completion_amount_capping_limit(setup_test_key_manager):
    app = create_app()
    mgr = setup_test_key_manager
    
    # Create a key with a small cap of $0.0001
    record = mgr.create_key(name="Small Budget User", token_cap=-1, amount_cap=0.0001)
    api_key = record.key
    
    # Mock model templates
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
    mock_tokenizer.eos_token_ids = [0]
    
    with TestClient(app) as client, \
         patch.object(model_wrapper, "model", MagicMock()), \
         patch.object(model_wrapper, "tokenizer", mock_tokenizer):
        
        # Call with correct header - should authenticate successfully
        def mock_add_request(request_ctx):
            request_ctx.completion_tokens = 5
            request_ctx.response_queue.put_nowait(("Hello ", None))
            request_ctx.response_queue.put_nowait(("World!", "stop"))

        with patch("app.api.routes.inference_scheduler.add_request", side_effect=mock_add_request):
            # Request under the cap
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": 10,
                    "stream": False
                }
            )
            assert response.status_code == 200
            
        # Simulate usage exceeding the cap (spending $0.00015 > $0.0001)
        mgr.record_usage(api_key, prompt_tokens=1000, completion_tokens=0) # 1000 tokens * 0.15/1M = 0.00015
        
        # Request after exceeding cap - should receive 403 Forbidden
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "messages": [{"role": "user", "content": "Hello again!"}],
                "max_tokens": 10,
                "stream": False
            }
        )
        assert response.status_code == 403
        assert "spent" in response.json()["detail"].lower()
