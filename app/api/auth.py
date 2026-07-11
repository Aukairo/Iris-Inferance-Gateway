from typing import Optional
from fastapi import Depends, HTTPException, Security, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config.config import settings
from app.core.logging import logger
from app.core.key_manager import key_manager

# Declare Bearer token auth helper (auto_error=False handles empty auth gracefully)
security = HTTPBearer(auto_error=False)


def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Dependency to verify the API Key provided in headers.
    If no keys are configured (either statically or dynamically), authentication is skipped.
    """
    request.state.api_key = None

    # Determine if authentication is active
    all_keys = key_manager.list_keys()
    has_active_keys = len(all_keys) > 0 or bool(settings.api_key)

    if not has_active_keys:
        return None

    if credentials is None:
        logger.warning("Authentication failed: missing Bearer token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing API key."
        )

    key = credentials.credentials
    record = key_manager.verify_key(key)

    if not record:
        logger.warning("Authentication failed: invalid Bearer token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid API key."
        )

    # Check cap
    if record.token_cap != -1 and record.total_tokens_used >= record.token_cap:
        logger.warning(
            "Key limit exceeded: %s (%d/%d tokens)",
            record.name,
            record.total_tokens_used,
            record.token_cap
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: API Key usage limit exceeded ({record.total_tokens_used}/{record.token_cap} tokens)."
        )

    if record.amount_cap != -1.0 and record.amount_spent >= record.amount_cap:
        logger.warning(
            "Key limit exceeded: %s ($%.4f/$%.4f spent)",
            record.name,
            record.amount_spent,
            record.amount_cap
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: API Key usage limit exceeded (${record.amount_spent:.4f}/${record.amount_cap:.4f} spent)."
        )

    # Save to request state for completions tracking
    request.state.api_key = key
    return key
