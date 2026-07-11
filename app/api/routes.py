import time
import uuid
import asyncio
import resource
from typing import Union, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config.config import settings
from app.core.logging import logger
from app.api.auth import verify_api_key
from app.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionUsage,
)
from app.inference.engine import model_wrapper
from app.batching.scheduler import inference_scheduler, RequestContext
from app.streaming.streamer import stream_chat_generator

router = APIRouter()
START_TIME = time.time()


@router.post(
    "/v1/chat/completions",
    response_model=Union[ChatCompletionResponse, Any],  # StreamingResponse or ChatCompletionResponse
    dependencies=[Depends(verify_api_key)]
)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """
    OpenAI-compatible Chat Completions endpoint.
    Supports streaming and request batching.
    """
    if model_wrapper.model is None or model_wrapper.tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on the server."
        )

    # Validate and tokenize the prompt messages using tokenizer chat template
    try:
        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        # apply_chat_template applies the prompt format and returns token ids
        prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=True,
            add_generation_prompt=True
        )
    except Exception as e:
        logger.error("Failed to tokenize chat completion messages: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to tokenize prompt: {str(e)}"
        )

    # Prepare parameters (fallback to config defaults)
    temperature = request.temperature if request.temperature is not None else settings.temperature
    top_p = request.top_p if request.top_p is not None else settings.top_p
    max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens

    request_id = uuid.uuid4().hex
    response_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Extract verified API key from request state for token tracking
    api_key = getattr(http_request.state, "api_key", None)

    # Create inference request context
    request_ctx = RequestContext(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_queue=response_queue,
        loop=loop,
        api_key=api_key
    )

    # Queue request for dynamic batch scheduler
    inference_scheduler.add_request(request_ctx)

    if request.stream:
        # Return Streaming Response (Server-Sent Events)
        return StreamingResponse(
            stream_chat_generator(request_ctx, model_wrapper.model_name or "mlx-model"),
            media_type="text/event-stream"
        )
    else:
        # Synchronous completion: wait for all tokens to be gathered
        full_text = ""
        finish_reason = "stop"
        try:
            while True:
                item = await response_queue.get()
                text_segment, reason = item
                if text_segment is not None:
                    full_text += text_segment
                if reason is not None or text_segment is None:
                    finish_reason = reason or "stop"
                    break
        except Exception as e:
            logger.error("Exception occurred while awaiting completion: %s", str(e))
            request_ctx.cancelled = True
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference execution failed: {str(e)}"
            )

        # Build final response payload
        prompt_len = len(prompt_tokens)
        completion_len = request_ctx.completion_tokens
        response_payload = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model_wrapper.model_name or "mlx-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(role="assistant", content=full_text),
                    finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len
            )
        )
        return response_payload


@router.get("/health")
async def health_check() -> dict:
    """
    Exposes health status, model loading, queue lengths, and memory statistics.
    """
    # ru_maxrss is in bytes on macOS
    max_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_mb = round(max_rss_bytes / (1024 * 1024), 2)
    
    uptime_sec = int(time.time() - START_TIME)
    
    return {
        "status": "healthy",
        "model_loaded": model_wrapper.model is not None,
        "model_name": model_wrapper.model_name,
        "queue_length": inference_scheduler.incoming_queue.qsize(),
        "active_generations": len(inference_scheduler.active_requests),
        "uptime_seconds": uptime_sec,
        "memory_usage_mb": max_rss_mb
    }


@router.get("/metrics")
async def metrics() -> Response:
    """
    Serves Prometheus metrics for telemetry monitoring.
    """
    if not settings.metrics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics collection is disabled."
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
