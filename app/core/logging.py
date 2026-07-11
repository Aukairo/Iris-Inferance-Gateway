import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict
from app.config.config import settings

class JSONFormatter(logging.Formatter):
    """
    Structured JSON formatter that formats log records into JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Inject extra attributes passed via logging.log(..., extra={...})
        extra_fields = [
            "request_id", "latency_ms", "model", "prompt_tokens", 
            "generated_tokens", "tokens_per_sec", "queue_size", 
            "batch_size"
        ]
        for field in extra_fields:
            if hasattr(record, field):
                log_payload[field] = getattr(record, field)

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload)


def setup_logging() -> logging.Logger:
    """
    Configures and returns the main application logger.
    """
    logger = logging.getLogger("mlx_server")
    logger.setLevel(getattr(logging, settings.logging_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    # Prevent propagation to root logger to avoid unstructured double-logging
    logger.propagate = False
    
    return logger

# Shared logger instance
logger = setup_logging()
