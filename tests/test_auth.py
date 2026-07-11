import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from app.config.config import settings
from app.api.auth import verify_api_key
from app.core.key_manager import KeyManager, ApiKeyRecord

TEST_KEYS_FILE = "data/test_auth_keys.json"

@pytest.fixture(autouse=True)
def setup_auth_test_key_manager():
    """
    Isolate key manager for auth tests.
    """
    import os
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)
        
    test_manager = KeyManager(filepath=TEST_KEYS_FILE)
    test_manager.init_store()
    
    with patch("app.api.auth.key_manager", test_manager):
        yield test_manager
        
    if os.path.exists(TEST_KEYS_FILE):
        os.remove(TEST_KEYS_FILE)


class MockRequest:
    def __init__(self):
        self.state = MagicMock()


def test_auth_disabled(monkeypatch: pytest.MonkeyPatch, setup_auth_test_key_manager) -> None:
    """
    If no keys are configured, verification should pass without credentials.
    """
    monkeypatch.setattr(settings, "api_key", None)
    req = MockRequest()
    # Should complete without raising any exception
    verify_api_key(req, None)
    assert req.state.api_key is None


def test_auth_enabled_valid_token(monkeypatch: pytest.MonkeyPatch, setup_auth_test_key_manager) -> None:
    """
    If settings.api_key is set, verification should pass with correct Bearer credentials.
    """
    monkeypatch.setattr(settings, "api_key", "secret-token")
    # Re-initialize store so static key gets registered
    setup_auth_test_key_manager.init_store()
    
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-token")
    req = MockRequest()
    # Should complete successfully
    verify_api_key(req, creds)
    assert req.state.api_key == "secret-token"


def test_auth_enabled_invalid_token(monkeypatch: pytest.MonkeyPatch, setup_auth_test_key_manager) -> None:
    """
    If settings.api_key is set, verification should raise 401 HTTP exception for incorrect token.
    """
    monkeypatch.setattr(settings, "api_key", "secret-token")
    setup_auth_test_key_manager.init_store()
    
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")
    req = MockRequest()
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(req, creds)
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in exc_info.value.detail


def test_auth_enabled_missing_token(monkeypatch: pytest.MonkeyPatch, setup_auth_test_key_manager) -> None:
    """
    If settings.api_key is set, verification should raise 401 HTTP exception if credentials are None.
    """
    monkeypatch.setattr(settings, "api_key", "secret-token")
    setup_auth_test_key_manager.init_store()
    
    req = MockRequest()
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(req, None)
    assert exc_info.value.status_code == 401
