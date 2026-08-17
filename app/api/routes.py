import time
import uuid
import asyncio
import resource
import json
from typing import Union, Any, List, Dict, Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config.config import settings
from app.core.logging import logger
from app.api.auth import verify_api_key
from app.inference.tool_handler import (
    format_tool_system_prompt,
    parse_tool_calls,
    extract_reasoning_content,
)
from app.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionUsage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    CompletionChunk,
    CompletionChunkChoice,
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


def normalize_messages(
    messages: List[Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Normalizes a list of ChatMessage objects for tokenizers with strict role alternation
    and content type checks (like Gemma-3):
    1. Extracts and keeps the system message at the beginning (or injects tool system prompt).
    2. Converts tool responses to 'user' role with formatted content including tool_call_id.
    3. Handles assistant tool_calls, formatting tool_calls into the content text.
    4. Merges consecutive messages of the same role to guarantee strict user/assistant alternation.
    5. Ensures content is always a valid string (never None).
    """
    if not messages:
        return []

    system_message = None
    processed = []

    # Map tool_call_id -> function name from assistant messages
    tool_call_name_map = {}
    for msg in messages:
        if getattr(msg, "role", None) == "assistant":
            tcs = getattr(msg, "tool_calls", None) or (msg.model_extra.get("tool_calls") if msg.model_extra else None)
            if tcs:
                for tc in tcs:
                    if isinstance(tc, dict):
                        cid = tc.get("id")
                        func = tc.get("function", {})
                        fname = func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
                    else:
                        cid = getattr(tc, "id", None)
                        func = getattr(tc, "function", {})
                        fname = getattr(func, "name", None) if not isinstance(func, dict) else func.get("name")
                    if cid and fname:
                        tool_call_name_map[cid] = fname

    for msg in messages:
        role = msg.role
        content = get_content_as_str(msg.content)

        tool_calls = getattr(msg, "tool_calls", None) or (msg.model_extra.get("tool_calls") if msg.model_extra else None)
        name = getattr(msg, "name", None) or (msg.model_extra.get("name") if msg.model_extra else None)
        tool_call_id = getattr(msg, "tool_call_id", None) or (msg.model_extra.get("tool_call_id") if msg.model_extra else None)

        if role == "system":
            system_message = {"role": "system", "content": content}
            continue

        if role == "assistant" and tool_calls:
            tool_calls_text = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                else:
                    func = getattr(tc, "function", {})

                if isinstance(func, dict):
                    name_str = func.get("name", "unknown")
                    args_str = func.get("arguments", "{}")
                else:
                    name_str = getattr(func, "name", "unknown")
                    args_str = getattr(func, "arguments", "{}")
                tool_calls_text.append(f"<tool_call>\n{{\"name\": \"{name_str}\", \"arguments\": {args_str}}}\n</tool_call>")

            calls_str = "\n".join(tool_calls_text)
            if content:
                content = f"{content}\n\n{calls_str}"
            else:
                content = calls_str

        if role == "tool":
            resolved_name = name or (tool_call_name_map.get(tool_call_id) if tool_call_id else None) or tool_call_id or "tool"
            if tool_call_id and (name or tool_call_name_map.get(tool_call_id)):
                content = f"[Tool Response for '{resolved_name}' (ID: {tool_call_id})]:\n{content}"
            else:
                content = f"[Tool Response for '{resolved_name}']:\n{content}"
            role = "user"

        if role not in ["user", "assistant"]:
            role = "user"

        processed.append({"role": role, "content": content})

    if tools and tool_choice != "none":
        tool_prompt = format_tool_system_prompt(tools, tool_choice=tool_choice)
        if tool_prompt:
            if system_message:
                system_message["content"] = f"{system_message['content']}\n{tool_prompt}"
            else:
                system_message = {"role": "system", "content": tool_prompt.strip()}

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

    if system_message:
        merged.insert(0, system_message)

    return merged


@router.post(
    "/v1/chat/completions",
    response_model=Union[ChatCompletionResponse, Any],
    dependencies=[Depends(verify_api_key)]
)
@router.post(
    "/chat/completions",
    response_model=Union[ChatCompletionResponse, Any],
    dependencies=[Depends(verify_api_key)]
)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """
    OpenAI-compatible Chat Completions endpoint.
    Supports streaming, multi-model tool calling, reasoning models, and request batching.
    """
    if model_wrapper.model is None or model_wrapper.tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on the server."
        )

    response_model_name = request.model or model_wrapper.model_name or "mlx-model"

    # Validate and tokenize the prompt messages using tokenizer chat template
    try:
        messages_dicts = []
        for m in request.messages:
            dict_msg = {"role": m.role}
            content = get_content_as_str(m.content)
            if content is not None:
                dict_msg["content"] = content

            tool_calls = getattr(m, "tool_calls", None) or (m.model_extra.get("tool_calls") if m.model_extra else None)
            name = getattr(m, "name", None) or (m.model_extra.get("name") if m.model_extra else None)
            tool_call_id = getattr(m, "tool_call_id", None) or (m.model_extra.get("tool_call_id") if m.model_extra else None)

            if tool_calls:
                formatted_tc = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        formatted_tc.append(tc)
                    else:
                        func = getattr(tc, "function", {})
                        func_name = getattr(func, "name", "") if not isinstance(func, dict) else func.get("name", "")
                        func_args = getattr(func, "arguments", "{}") if not isinstance(func, dict) else func.get("arguments", "{}")
                        formatted_tc.append({
                            "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:12]}"),
                            "type": "function",
                            "function": {"name": func_name, "arguments": func_args}
                        })
                dict_msg["tool_calls"] = formatted_tc
            if name:
                dict_msg["name"] = name
            if tool_call_id:
                dict_msg["tool_call_id"] = tool_call_id

            if "content" not in dict_msg:
                dict_msg["content"] = None

            messages_dicts.append(dict_msg)

        # Support response_format: {"type": "json_object"}
        if request.response_format and request.response_format.get("type") == "json_object":
            has_system = False
            for dm in messages_dicts:
                if dm.get("role") == "system":
                    dm["content"] = (dm["content"] or "") + "\n\nCRITICAL: Respond strictly in valid JSON format."
                    has_system = True
                    break
            if not has_system:
                messages_dicts.insert(0, {"role": "system", "content": "CRITICAL: Respond strictly in valid JSON format."})

        has_tools = bool(request.tools) and (request.tool_choice != "none")

        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if has_tools:
            try:
                prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
                    messages_dicts,
                    tools=request.tools,
                    **kwargs
                )
            except TypeError:
                messages_dicts = normalize_messages(request.messages, tools=request.tools, tool_choice=request.tool_choice)
                prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
                    messages_dicts,
                    **kwargs
                )
        else:
            prompt_tokens = model_wrapper.tokenizer.apply_chat_template(
                messages_dicts,
                **kwargs
            )
    except Exception as raw_err:
        logger.warning(
            "Tokenizer failed to apply native chat template: %s. Falling back to role normalization.",
            str(raw_err)
        )
        try:
            messages_dicts = normalize_messages(request.messages, tools=request.tools, tool_choice=request.tool_choice)
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

    temperature = request.temperature if request.temperature is not None else settings.temperature
    top_p = request.top_p if request.top_p is not None else settings.top_p
    max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens

    request_id = uuid.uuid4().hex
    response_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    api_key = getattr(http_request.state, "api_key", None)

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

    inference_scheduler.add_request(request_ctx)

    if request.stream:
        return StreamingResponse(
            stream_chat_generator(
                request_ctx,
                response_model_name,
                has_tools=has_tools
            ),
            media_type="text/event-stream"
        )
    else:
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
                        model=response_model_name,
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

        prompt_len = len(prompt_tokens)
        completion_len = request_ctx.completion_tokens

        reasoning_content, clean_text = extract_reasoning_content(full_text)

        if has_tools:
            clean_content, tool_calls, parsed_finish_reason = parse_tool_calls(clean_text)
            if tool_calls:
                message_obj = ChatCompletionChoiceMessage(
                    role="assistant",
                    content=clean_content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls
                )
                finish_reason = "tool_calls"
            else:
                message_obj = ChatCompletionChoiceMessage(
                    role="assistant",
                    content=clean_text,
                    reasoning_content=reasoning_content
                )
        else:
            message_obj = ChatCompletionChoiceMessage(
                role="assistant",
                content=clean_text,
                reasoning_content=reasoning_content
            )

        response_payload = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=response_model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message_obj,
                    finish_reason=finish_reason
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len
            )
        )

        try:
            from app.core.mongodb import mongodb_manager
            latency_ms = int((time.perf_counter() - request_ctx.time_queued) * 1000)
            asyncio.create_task(
                mongodb_manager.save_request_log(
                    request_id=request_id,
                    api_key=api_key,
                    model=response_model_name,
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


@router.post(
    "/v1/completions",
    response_model=Union[CompletionResponse, Any],
    dependencies=[Depends(verify_api_key)]
)
@router.post(
    "/completions",
    response_model=Union[CompletionResponse, Any],
    dependencies=[Depends(verify_api_key)]
)
async def text_completions(
    request: CompletionRequest,
    http_request: Request
) -> Union[CompletionResponse, StreamingResponse]:
    """
    OpenAI-compatible Legacy Text Completions endpoint.
    Supports streaming and text generation for arbitrary prompt strings or token sequences.
    """
    if model_wrapper.model is None or model_wrapper.tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on the server."
        )

    response_model_name = request.model or model_wrapper.model_name or "mlx-model"

    raw_prompt = request.prompt
    if isinstance(raw_prompt, list):
        if raw_prompt and isinstance(raw_prompt[0], int):
            prompt_tokens = raw_prompt
        elif raw_prompt and isinstance(raw_prompt[0], str):
            prompt_tokens = model_wrapper.tokenizer.encode(raw_prompt[0])
        else:
            prompt_tokens = model_wrapper.tokenizer.encode(str(raw_prompt))
    else:
        prompt_tokens = model_wrapper.tokenizer.encode(str(raw_prompt))

    temperature = request.temperature if request.temperature is not None else settings.temperature
    top_p = request.top_p if request.top_p is not None else settings.top_p
    max_tokens = request.max_tokens if request.max_tokens is not None else settings.max_tokens

    request_id = uuid.uuid4().hex
    response_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    api_key = getattr(http_request.state, "api_key", None)

    request_ctx = RequestContext(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_queue=response_queue,
        loop=loop,
        api_key=api_key,
        messages=[{"role": "user", "content": str(raw_prompt)}]
    )

    inference_scheduler.add_request(request_ctx)

    if request.stream:
        async def stream_completion_generator() -> AsyncGenerator[str, None]:
            created_time = int(time.time())
            while True:
                try:
                    item = await asyncio.wait_for(request_ctx.response_queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'error': {'message': 'Generation timeout'}})}\n\n"
                    break

                text_segment, reason = item
                if text_segment is not None:
                    chunk = CompletionChunk(
                        id=f"cmpl-{request_id}",
                        created=created_time,
                        model=response_model_name,
                        choices=[
                            CompletionChunkChoice(
                                index=0,
                                text=text_segment,
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                if reason is not None or text_segment is None:
                    final_chunk = CompletionChunk(
                        id=f"cmpl-{request_id}",
                        created=created_time,
                        model=response_model_name,
                        choices=[
                            CompletionChunkChoice(
                                index=0,
                                text="",
                                finish_reason=reason or "stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                    break
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_completion_generator(), media_type="text/event-stream")
    else:
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
            request_ctx.cancelled = True
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference execution failed: {str(e)}"
            )

        prompt_len = len(prompt_tokens)
        completion_len = request_ctx.completion_tokens

        return CompletionResponse(
            id=f"cmpl-{request_id}",
            created=int(time.time()),
            model=response_model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=full_text,
                    finish_reason=finish_reason
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_len,
                completion_tokens=completion_len,
                total_tokens=prompt_len + completion_len
            )
        )


@router.get("/health")
async def health_check() -> dict:
    """
    Exposes health status, model loading, queue lengths, and memory statistics.
    """
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
                created=int(START_TIME),
                owned_by="iris-engine"
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
    Returns ModelInfo for any requested model string to support custom client configurations.
    """
    return ModelInfo(
        id=model_id,
        created=int(START_TIME),
        owned_by="iris-engine"
    )
