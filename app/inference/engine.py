import time
from typing import Tuple, Any, Optional
import mlx.core as mx  # type: ignore
from mlx_lm import load
from app.config.config import settings
from app.core.logging import logger


class ModelWrapper:
    """
    A singleton-like wrapper for loading and maintaining the MLX model and tokenizer in memory.
    """
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_name: Optional[str] = None

    def load_model(self, model_name: str) -> None:
        """
        Loads the specified MLX model and tokenizer into unified memory.
        Uses MLX/Metal acceleration automatically on Apple Silicon.
        """
        if self.model is not None:
            logger.warning(
                "Model already loaded (%s). Skipping reload request.",
                self.model_name
            )
            return

        logger.info("Initializing MLX inference engine with model: %s", model_name)
        t_start = time.perf_counter()
        try:
            try:
                loaded = load(model_name)
                self.model = loaded[0]
                self.tokenizer = loaded[1]
                self.processor = getattr(loaded, "processor", getattr(self.tokenizer, "processor", None))
            except Exception as lm_err:
                # Fallback to mlx_vlm for Vision-Language Models
                try:
                    from mlx_vlm import load as load_vlm
                    logger.info("Attempting load via mlx_vlm for vision-language model...")
                    model_vlm, processor_vlm = load_vlm(model_name)
                    self.model = model_vlm
                    self.processor = processor_vlm
                    self.tokenizer = getattr(processor_vlm, "tokenizer", processor_vlm)
                except Exception as vlm_err:
                    logger.error("Failed loading with mlx_lm (%s) and mlx_vlm (%s)", str(lm_err), str(vlm_err))
                    raise lm_err

            self.model_name = model_name
            
            # Print GPU details if available
            if mx.metal.is_available():
                logger.info("Metal GPU acceleration enabled for MLX.")
            else:
                logger.warning("Metal acceleration is not available; falling back to CPU.")
                
            t_duration = time.perf_counter() - t_start
            logger.info("Model loaded successfully in %.2f seconds", t_duration)
        except Exception as e:
            logger.error("Failed to load model %s: %s", model_name, str(e), exc_info=True)
            raise e

    def unload_model(self) -> None:
        """
        Clears model weights from memory.
        mx.clear_cache() is dispatched to a daemon thread — calling it on the
        main thread while GPU ops may still be in-flight (from the worker
        daemon thread being reaped) can block indefinitely on the Metal stream.
        """
        import threading
        self.model = None
        self.tokenizer = None
        self.model_name = None

        def _clear():
            try:
                mx.clear_cache()
            except Exception:
                pass

        threading.Thread(target=_clear, daemon=True).start()
        logger.info("MLX model cleared from memory.")



# Singleton instance
model_wrapper = ModelWrapper()
