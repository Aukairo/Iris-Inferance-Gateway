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
            # Timeout check ensures we don't block indefinitely if the queue gets stuck
            try:
                item = await asyncio.wait_for(request_ctx.response_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for tokens on request %s", request_id)
                yield "data: [ERROR] Generation timeout\n\n"
                break

            text_segment, finish_reason = item

            if text_segment is not None:
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
        raise

    except Exception as e:
        logger.error("Exception in streaming generator for request %s: %s", request_id, str(e), exc_info=True)
        yield f"data: {json.dumps({'error': {'message': 'Generation error: ' + str(e)}})}\n\n"
        yield "data: [DONE]\n\n"
