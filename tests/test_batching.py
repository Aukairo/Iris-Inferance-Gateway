import asyncio
import pytest
from app.batching.scheduler import RequestContext, InferenceScheduler


@pytest.mark.asyncio
async def test_request_context_initialization() -> None:
    """
    Verify that RequestContext correctly stores metadata, loop references, and timing.
    """
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    request_id = "test-req-123"
    prompt = [101, 2054, 2003, 1037]

    request = RequestContext(
        request_id=request_id,
        prompt_tokens=prompt,
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
        response_queue=queue,
        loop=loop
    )

    assert request.request_id == request_id
    assert request.prompt_tokens == prompt
    assert request.max_tokens == 50
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.response_queue is queue
    assert request.loop is loop
    assert request.cancelled is False
    assert request.time_queued > 0
    assert request.time_first_token is None
    assert request.completion_tokens == 0


def test_scheduler_add_request() -> None:
    """
    Verify that InferenceScheduler adds requests to its queue and tracks queue metrics.
    """
    scheduler = InferenceScheduler()
    # Create mock contexts
    req1 = RequestContext("id1", [1], 10, 0.7, 0.9, None, None)
    req2 = RequestContext("id2", [2], 20, 0.7, 0.9, None, None)

    assert scheduler.incoming_queue.qsize() == 0

    scheduler.add_request(req1)
    assert scheduler.incoming_queue.qsize() == 1

    scheduler.add_request(req2)
    assert scheduler.incoming_queue.qsize() == 2

    # Pull items from queue to verify order
    pulled1 = scheduler.incoming_queue.get_nowait()
    assert pulled1.request_id == "id1"
    
    pulled2 = scheduler.incoming_queue.get_nowait()
    assert pulled2.request_id == "id2"
    
    assert scheduler.incoming_queue.empty()
