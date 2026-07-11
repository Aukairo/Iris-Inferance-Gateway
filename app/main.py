import time
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config.config import settings
from app.core.logging import logger
from app.api.routes import router
from app.api.admin_routes import router as admin_router
from app.core.key_manager import key_manager
from app.core.config_manager import config_manager
from app.inference.engine import model_wrapper
from app.batching.scheduler import inference_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles application startup and shutdown lifespans.
    Pre-loads the configured MLX model once, and runs the inference worker thread.
    """
    logger.info("Starting MLX local inference server...")
    
    # 0. Initialize persistent configuration & API key databases
    try:
        config_manager.init_store()
    except Exception as e:
        logger.error("Failed to initialize configuration database: %s", str(e))

    try:
        key_manager.init_store()
    except Exception as e:
        logger.error("Failed to initialize API key database: %s", str(e))
    
    # 1. Load MLX Model exactly once
    try:
        model_wrapper.load_model(settings.model_name)
    except Exception as e:
        logger.critical(
            "Fatal: Failed to load model %s. Server shutting down.",
            settings.model_name,
            exc_info=True
        )
        raise e

    # 2. Start the Inference Worker background thread
    inference_scheduler.start()
    
    logger.info("Application startup sequence completed. Ready to serve requests.")
    yield

    # 3. Shutdown: Clean up background threads and memory
    logger.info("Shutting down application server...")
    
    # Stop public sharing tunnel if active
    logger.info("[shutdown] step 1/3: stopping tunnel...")
    try:
        from app.core.tunnel_manager import tunnel_manager
        import threading
        
        # Stop tunnel in a separate thread with timeout to prevent hanging
        def stop_tunnel_with_timeout():
            try:
                tunnel_manager.stop_tunnel()
            except Exception as e:
                logger.error(f"Tunnel stop error: {e}")
        
        tunnel_thread = threading.Thread(target=stop_tunnel_with_timeout, daemon=True)
        tunnel_thread.start()
        tunnel_thread.join(timeout=3.0)  # Wait max 3 seconds
        
        if tunnel_thread.is_alive():
            logger.warning("Tunnel stop timeout - continuing shutdown anyway")
        else:
            logger.info("[shutdown] step 1/3: tunnel stopped.")
    except Exception as e:
        logger.error(f"Error stopping sharing tunnel: {str(e)}")
    
    logger.info("[shutdown] step 2/3: stopping inference scheduler...")
    inference_scheduler.stop()
    logger.info("[shutdown] step 2/3: scheduler stopped.")

    logger.info("[shutdown] step 3/3: unloading model...")
    model_wrapper.unload_model()
    logger.info("[shutdown] step 3/3: model unloaded.")

    logger.info("Clean shutdown completed.")


def create_app() -> FastAPI:
    """
    Factory function to initialize the FastAPI application.
    """
    app = FastAPI(
        title="Iris Engine",
        description="Production-quality LLM inference server optimized for Apple Silicon using MLX",
        version="1.0.0",
        lifespan=lifespan
    )

    # Enable Cross-Origin Resource Sharing (CORS) for frontends/clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global request duration middleware for logging
    @app.middleware("http")
    async def log_request_metrics(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Avoid logging metrics endpoint calls
        if request.url.path not in ["/metrics", "/health"]:
            logger.info(
                "Request processed: %s %s - Status: %d - Duration: %dms",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms
            )
        return response

    # Mount completions and default API routing
    app.include_router(router)

    # Mount API Key Management & admin routes
    app.include_router(admin_router)

    # Mount Admin Dashboard UI route
    @app.get("/admin", response_class=HTMLResponse, tags=["admin"])
    async def get_admin_dashboard():
        admin_html_path = os.path.join(os.path.dirname(__file__), "static", "admin.html")
        if not os.path.exists(admin_html_path):
            raise HTTPException(status_code=404, detail="Admin HTML page not found.")
        with open(admin_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return app


# Main entry point instance
app = FastAPI()
app = create_app()
