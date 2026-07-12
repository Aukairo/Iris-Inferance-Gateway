import json
import time
import asyncio
from typing import AsyncGenerator, Dict, Any
from app.core.logging import logger
from app.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from app.batching.scheduler import RequestContext


async def stream_chat_generator(
    request_ctx: RequestContext,
    model_name: str
) -> AsyncGenerator[str, None]:
    """
    Asynchronously yields Server-Sent Events (SSE) chat completion chunks.
    Monitors for client disconnects and triggers cleanups.
    """
    request_id = request_ctx.request_id
    created_time = int(time.time())
    choice_index = 0
    full_response = ""
    start_time = time.perf_counter()
    status = "success"
    error_msg = None

    try:
        # Yield the initial chunk containing the role definition
        initial_chunk = ChatCompletionChunk(
            id=f"chatcmpl-{request_id}",
            created=created_time,
            model=model_name,
            choices=[
                ChatCompletionChunkChoice(
                    index=choice_index,
                    delta=ChatCompletionChunkDelta(role="assistant"),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

        while True:
            # Wait for next token from the queue
            try:
                item = await asyncio.wait_for(request_ctx.response_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for tokens on request %s", request_id)
                yield f"data: {json.dumps({'error': {'message': 'Generation timeout'}})}\n\n"
                status = "error"
                error_msg = "Generation timeout"
                break

            text_segment, finish_reason = item

            if text_segment is not None:
                full_response += text_segment
                # Yield text segment chunk
                chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{request_id}",
                    created=created_time,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=choice_index,
                            delta=ChatCompletionChunkDelta(content=text_segment),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

            if finish_reason is not None or text_segment is None:
                # Sequence completed or terminated
                final_chunk = ChatCompletionChunk(
                    id=f"chatcmpl-{request_id}",
                    created=created_time,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=choice_index,
                            delta=ChatCompletionChunkDelta(content=""),
                            finish_reason=finish_reason or "stop"
                        )
                    ]
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                break

        # Send standard OpenAI termination message
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        # Caught when client terminates the request early or HTTP connection drops
        logger.warning("Streaming request %s cancelled by client/network disconnect.", request_id)
        request_ctx.cancelled = True
        status = "cancelled"
        raise

    except Exception as e:
        logger.error("Exception in streaming generator for request %s: %s", request_id, str(e), exc_info=True)
        status = "error"
        error_msg = str(e)
        yield f"data: {json.dumps({'error': {'message': 'Generation error: ' + str(e)}})}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        try:
            from app.core.mongodb import mongodb_manager
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            asyncio.create_task(
                mongodb_manager.save_request_log(
                    request_id=request_id,
                    api_key=request_ctx.api_key,
                    model=model_name,
                    prompt=request_ctx.messages,
                    response=full_response,
                    prompt_tokens=len(request_ctx.prompt_tokens),
                    completion_tokens=request_ctx.completion_tokens,
                    cached_tokens=request_ctx.cached_tokens,
                    latency_ms=latency_ms,
                    stream=True,
                    status=status,
                    error_message=error_msg
                )
            )
        except Exception as log_err:
            logger.error("Error logging streaming request success/failure: %s", str(log_err))
