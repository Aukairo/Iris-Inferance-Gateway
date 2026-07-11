from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from typing import List, Optional
from app.config.config import settings
from app.core.key_manager import key_manager, ApiKeyRecord
from app.core.config_manager import config_manager, ModelPricingConfig
from app.core.tunnel_manager import tunnel_manager

router = APIRouter(prefix="/api/admin", tags=["admin"])

async def verify_admin_password(request: Request) -> None:
    """
    Validates that the request comes from an authenticated administrator.
    Checks X-Admin-Password or Authorization: Bearer <password>.
    """
    if not settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin interface is disabled because ADMIN_PASSWORD is not configured."
        )
        
    admin_pass = request.headers.get("X-Admin-Password")
    if not admin_pass:
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            admin_pass = auth.split(" ")[1]

    if admin_pass != settings.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Admin Password."
        )


class CreateKeyRequest(BaseModel):
    name: str
    token_cap: int = -1
    amount_cap: float = -1.0


class UpdateCapRequest(BaseModel):
    token_cap: int
    amount_cap: float = -1.0


@router.get("/keys", response_model=List[ApiKeyRecord], dependencies=[Depends(verify_admin_password)])
async def get_keys():
    """
    Returns list of all registered API keys and their usage.
    """
    return key_manager.list_keys()


@router.post("/keys", response_model=ApiKeyRecord, dependencies=[Depends(verify_admin_password)])
async def create_key(req: CreateKeyRequest):
    """
    Generates a new API key.
    """
    if not req.name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Key name cannot be empty."
        )
    return key_manager.create_key(name=req.name, token_cap=req.token_cap, amount_cap=req.amount_cap)


@router.delete("/keys/{key_val:path}", dependencies=[Depends(verify_admin_password)])
async def delete_key(key_val: str):
    """
    Deletes/revokes an API key.
    """
    success = key_manager.delete_key(key_val)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found."
        )
    return {"status": "success", "message": "Key revoked successfully"}


@router.put("/keys/{key_val:path}/cap", dependencies=[Depends(verify_admin_password)])
async def update_cap(key_val: str, req: UpdateCapRequest):
    """
    Updates the token and budget caps for an API key.
    """
    success = key_manager.update_key_cap(key_val, req.token_cap, req.amount_cap)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found."
        )
    return {"status": "success", "message": "Token cap updated successfully"}


@router.post("/keys/{key_val:path}/rotate", dependencies=[Depends(verify_admin_password)])
async def rotate_api_key(key_val: str):
    """
    Rotates the API key bearer token for a client, keeping usage stats intact.
    """
    new_key = key_manager.rotate_key(key_val)
    if not new_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found."
        )
    return {"status": "success", "new_key": new_key, "message": "API key rotated successfully"}


@router.post("/keys/{key_val:path}/reset", dependencies=[Depends(verify_admin_password)])
async def reset_usage(key_val: str):
    """
    Resets token usage counters and cost spent back to zero.
    """
    success = key_manager.reset_key_usage(key_val)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found."
        )
    return {"status": "success", "message": "Token usage reset successfully"}


@router.get("/config/pricing", response_model=ModelPricingConfig, dependencies=[Depends(verify_admin_password)])
async def get_pricing_config():
    """
    Returns the current active model pricing configuration.
    """
    return config_manager.get_pricing()


@router.put("/config/pricing", response_model=ModelPricingConfig, dependencies=[Depends(verify_admin_password)])
async def update_pricing_config(req: ModelPricingConfig):
    """
    Updates and persists the model pricing configuration.
    """
    return config_manager.update_pricing(
        price_input=req.price_per_1m_input_tokens,
        price_output=req.price_per_1m_output_tokens,
        price_cached=req.price_per_1m_cached_tokens
    )



@router.get("/tunnel/status", dependencies=[Depends(verify_admin_password)])
async def get_tunnel_status():
    """
    Returns the current status of the sharing tunnel.
    """
    return tunnel_manager.get_status()


@router.post("/tunnel/start", dependencies=[Depends(verify_admin_password)])
async def start_tunnel_connection():
    """
    Initiates the background sharing tunnel.
    """
    return tunnel_manager.start_tunnel()


@router.post("/tunnel/stop", dependencies=[Depends(verify_admin_password)])
async def stop_tunnel_connection():
    """
    Terminates the sharing tunnel connection.
    """
    return tunnel_manager.stop_tunnel()

