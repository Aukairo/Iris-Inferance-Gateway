import json
import time
import asyncio
from typing import AsyncGenerator, Dict, Any
from app.core.logging import logger
from app.inference.tool_handler import parse_tool_calls, extract_reasoning_content
from app.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaFunctionCall,
)
from app.batching.scheduler import RequestContext


async def stream_chat_generator(
    request_ctx: RequestContext,
    model_name: str,
    has_tools: bool = False
) -> AsyncGenerator[str, None]:
    """
    Asynchronously yields Server-Sent Events (SSE) chat completion chunks.
    Monitors for client disconnects and triggers cleanups. Supports streaming tool calls and reasoning content.
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

        buffered_text = ""
        in_tool_block = False

        while True:
            # Wait for next token from the queue (allow up to 300s for large prompt prefilling)
            try:
                item = await asyncio.wait_for(request_ctx.response_queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for tokens on request %s", request_id)
                yield f"data: {json.dumps({'error': {'message': 'Generation timeout'}})}\n\n"
                status = "error"
                error_msg = "Generation timeout"
                break

            text_segment, finish_reason = item

            if text_segment is not None:
                full_response += text_segment
                if has_tools:
                    buffered_text += text_segment
                    lower_buf = buffered_text.lower()
                    if not in_tool_block:
                        if "<tool_call>" in lower_buf or "<call:" in lower_buf or "[tool_calls]" in lower_buf or "```json" in lower_buf:
                            in_tool_block = True
                            tag_pos = len(buffered_text)
                            for pattern in ["<tool_call>", "<call:", "[tool_calls]", "```json"]:
                                idx = lower_buf.find(pattern)
                                if idx != -1 and idx < tag_pos:
                                    tag_pos = idx

                            prefix_text = buffered_text[:tag_pos]
                            buffered_text = buffered_text[tag_pos:]
                            if prefix_text:
                                chunk = ChatCompletionChunk(
                                    id=f"chatcmpl-{request_id}",
                                    created=created_time,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=choice_index,
                                            delta=ChatCompletionChunkDelta(content=prefix_text),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        else:
                            # Keep a small safety tail (15 chars) in case a tool tag is cut across token chunks
                            if len(buffered_text) > 15:
                                to_yield = buffered_text[:-15]
                                buffered_text = buffered_text[-15:]
                                chunk = ChatCompletionChunk(
                                    id=f"chatcmpl-{request_id}",
                                    created=created_time,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=choice_index,
                                            delta=ChatCompletionChunkDelta(content=to_yield),
                                            finish_reason=None
                                        )
                                    ]
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                else:
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
                reasoning_content, clean_text_response = extract_reasoning_content(full_response)

                if has_tools:
                    clean_content, tool_calls, parsed_finish_reason = parse_tool_calls(clean_text_response)

                    if tool_calls:
                        delta_tool_calls = [
                            ChoiceDeltaToolCall(
                                index=i,
                                id=tc.id,
                                type="function",
                                function=ChoiceDeltaFunctionCall(
                                    name=tc.function.name,
                                    arguments=tc.function.arguments
                                )
                            )
                            for i, tc in enumerate(tool_calls)
                        ]

                        final_chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{request_id}",
                            created=created_time,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=choice_index,
                                    delta=ChatCompletionChunkDelta(
                                        reasoning_content=reasoning_content,
                                        tool_calls=delta_tool_calls
                                    ),
                                    finish_reason="tool_calls"
                                )
                            ]
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                    else:
                        if buffered_text:
                            # Flush remaining buffer
                            content_chunk = ChatCompletionChunk(
                                id=f"chatcmpl-{request_id}",
                                created=created_time,
                                model=model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=choice_index,
                                        delta=ChatCompletionChunkDelta(content=buffered_text),
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

                        final_chunk = ChatCompletionChunk(
                            id=f"chatcmpl-{request_id}",
                            created=created_time,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=choice_index,
                                    delta=ChatCompletionChunkDelta(reasoning_content=reasoning_content),
                                    finish_reason=finish_reason or "stop"
                                )
                            ]
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                else:
                    final_chunk = ChatCompletionChunk(
                        id=f"chatcmpl-{request_id}",
                        created=created_time,
                        model=model_name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=choice_index,
                                delta=ChatCompletionChunkDelta(reasoning_content=reasoning_content if "<think>" in full_response or "<thought>" in full_response else None),
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
