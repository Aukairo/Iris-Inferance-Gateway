import time
import uuid
import asyncio
import resource
from typing import Union, Any, List, Dict
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
    ModelInfo,
    ModelList,
)
from app.inference.engine import model_wrapper
from app.batching.scheduler import inference_scheduler, RequestContext
from app.streaming.streamer import stream_chat_generator

router = APIRouter()
START_TIME = time.time()


def get_content_as_str(content: Any) -> str:
    """
    Extract and concatenate text from raw message content.
    Handles standard string content and structured lists of content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def normalize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalizes a list of ChatMessage objects for tokenizers with strict role alternation
    and content type checks (like Gemma-3):
    1. Extracts and keeps the system message at the beginning.
    2. Converts tool responses to 'user' role with formatted content.
    3. Handles assistant tool_calls, formatting tool_calls into the content text.
    4. Merges consecutive messages of the same role to guarantee strict user/assistant alternation.
    5. Ensures content is always a valid string (never None).
    """
    if not messages:
        return []

    system_message = None
    processed = []

    for msg in messages:
        role = msg.role
        content = get_content_as_str(msg.content)
        
        # Get extra attributes (in case of tool_calls or name)
        tool_calls = getattr(msg, "tool_calls", None) or (msg.model_extra.get("tool_calls") if msg.model_extra else None)
        name = getattr(msg, "name", None) or (msg.model_extra.get("name") if msg.model_extra else None)

        # Extract system message
        if role == "system":
            system_message = {"role": "system", "content": content}
            continue

        # Format assistant tool calls into the content text if present
        if role == "assistant" and tool_calls:
            tool_calls_text = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                else:
                    func = getattr(tc, "function", {})

                if isinstance(func, dict):
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "{}")
                else:
                    name = getattr(func, "name", "unknown")
                    args = getattr(func, "arguments", "{}")
                tool_calls_text.append(f"[Call Tool: {name}({args})]")
            
            calls_str = "\n".join(tool_calls_text)
            if content:
                content = f"{content}\n\n{calls_str}"
            else:
                content = calls_str

        # Format tool response messages as user role
        if role == "tool":
            tool_name = name or "tool"
            content = f"[Tool Response for '{tool_name}']:\n{content}"
            role = "user"

        # Safe fallback for any other roles
        if role not in ["user", "assistant"]:
            role = "user"

        processed.append({"role": role, "content": content})

    # Merge consecutive messages of the same role
    merged = []
    for msg in processed:
        if not merged:
            merged.append(msg)
        else:
            last = merged[-1]
            if last["role"] == msg["role"]:
                last["content"] = f"{last['content']}\n\n{msg['content']}"
            else:
                merged.append(msg)

    # Re-insert the system message at the start if it existed
    if system_message:
        merged.insert(0, system_message)

    return merged


@router.post(
    "/v1/chat/completions",
    response_model=Union[ChatCompletionResponse, Any],  # StreamingResponse or ChatCompletionResponse
    dependencies=[Depends(verify_api_key)]
)
@router.post(
    "/chat/completions",
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
    # First, try to apply the template natively (needed for models supporting native tool calls)
    try:
        messages_dicts = []
        for m in request.messages:
            dict_msg = {"role": m.role}
            content = get_content_as_str(m.content)
            if content is not None:
                dict_msg["content"] = content
            
            tool_calls = getattr(m, "tool_calls", None) or (m.model_extra.get("tool_calls") if m.model_extra else None)
            name = getattr(m, "name", None) or (m.model_extra.get("name") if m.model_extra else None)
            if tool_calls:
                dict_msg["tool_calls"] = tool_calls
            if name:
                dict_msg["name"] = name
            
            if "content" not in dict_msg:
                dict_msg["content"] = None
                
            messages_dicts.append(dict_msg)

        prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=True,
            add_generation_prompt=True
        )
    except Exception as raw_err:
        logger.warning(
            "Tokenizer failed to apply native chat template: %s. Falling back to role normalization.",
            str(raw_err)
        )
        try:
            # Fallback: Normalize messages to guarantee strict user/assistant alternation and text content compatibility
            messages_dicts = normalize_messages(request.messages)
            prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
                messages_dicts,
                tokenize=True,
                add_generation_prompt=True
            )
        except Exception as norm_err:
            logger.error("Failed to tokenize chat completion messages even after normalization: %s", str(norm_err))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to tokenize prompt: {str(norm_err)}"
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
        api_key=api_key,
        messages=messages_dicts
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
            try:
                from app.core.mongodb import mongodb_manager
                latency_ms = int((time.perf_counter() - request_ctx.time_queued) * 1000)
                asyncio.create_task(
                    mongodb_manager.save_request_log(
                        request_id=request_id,
                        api_key=api_key,
                        model=model_wrapper.model_name or "mlx-model",
                        prompt=messages_dicts,
                        response=None,
                        prompt_tokens=len(prompt_tokens),
                        completion_tokens=request_ctx.completion_tokens,
                        cached_tokens=request_ctx.cached_tokens,
                        latency_ms=latency_ms,
                        stream=False,
                        status="error",
                        error_message=str(e)
                    )
                )
            except Exception as log_err:
                logger.error("Error logging request failure: %s", str(log_err))
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

        # Log successful completion to MongoDB in background
        try:
            from app.core.mongodb import mongodb_manager
            latency_ms = int((time.perf_counter() - request_ctx.time_queued) * 1000)
            asyncio.create_task(
                mongodb_manager.save_request_log(
                    request_id=request_id,
                    api_key=api_key,
                    model=model_wrapper.model_name or "mlx-model",
                    prompt=messages_dicts,
                    response=full_text,
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_len,
                    cached_tokens=request_ctx.cached_tokens,
                    latency_ms=latency_ms,
                    stream=False,
                    status="success"
                )
            )
        except Exception as log_err:
            logger.error("Error logging request success: %s", str(log_err))

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


@router.get(
    "/v1/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)]
)
@router.get(
    "/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)]
)
async def list_models() -> ModelList:
    """
    OpenAI-compatible models list endpoint.
    Returns the currently loaded or configured model.
    """
    active_model = model_wrapper.model_name or settings.model_name or "mlx-model"
    return ModelList(
        data=[
            ModelInfo(
                id=active_model,
                created=1686935002,
                owned_by="mlx-server"
            )
        ]
    )


@router.get(
    "/v1/models/{model_id:path}",
    response_model=ModelInfo,
    dependencies=[Depends(verify_api_key)]
)
@router.get(
    "/models/{model_id:path}",
    response_model=ModelInfo,
    dependencies=[Depends(verify_api_key)]
)
async def get_model(model_id: str) -> ModelInfo:
    """
    OpenAI-compatible model retrieve endpoint.
    """
    active_model = model_wrapper.model_name or settings.model_name or "mlx-model"
    if model_id != active_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found. Currently loaded model is '{active_model}'."
        )
    return ModelInfo(
        id=active_model,
        created=1686935002,
        owned_by="mlx-server"
    )
