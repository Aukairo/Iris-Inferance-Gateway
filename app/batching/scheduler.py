import time
import asyncio
from queue import Queue, Empty
from threading import Thread
from typing import Dict, List, Optional, Tuple, Any
import mlx.core as mx  # type: ignore
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

from app.config.config import settings
from app.core.logging import logger
from app.inference.engine import model_wrapper
from app.metrics import prometheus as pm
from app.core.key_manager import key_manager


class DynamicSampler:
    """
    A custom sampler that applies request-specific sampling parameters (temperature, top_p)
    to individual rows in the batched logprobs tensor.
    """
    def __init__(self, scheduler: "InferenceScheduler") -> None:
        self.scheduler = scheduler

    def __call__(self, logprobs: mx.array) -> mx.array:
        bg = self.scheduler.batch_generator
        if bg is None:
            return mx.argmax(logprobs, axis=-1)

        n_rows = logprobs.shape[0]
        uids: List[int] = []

        # Determine the UIDs for the current batch rows
        if bg.active_batch is not None and len(bg.active_batch.uids) == n_rows:
            uids = bg.active_batch.uids
        else:
            # During prompt processing, the rows correspond to unprocessed prompts
            num_prompts = min(len(bg.unprocessed_prompts), bg.prefill_batch_size)
            uids = [bg.unprocessed_prompts[i][0] for i in range(num_prompts)]

        sampled_tokens: List[mx.array] = []
        for i in range(n_rows):
            uid = uids[i] if i < len(uids) else None
            req = self.scheduler.active_requests.get(uid) if uid is not None else None

            # Get row logprobs (shape: 1, vocab_size)
            row_logprobs = logprobs[i : i + 1]

            if req and req.temperature == 0.0:
                sampled = mx.argmax(row_logprobs, axis=-1)
            elif req:
                # Apply custom request-level sampler
                sampler = make_sampler(req.temperature, top_p=req.top_p)
                sampled = sampler(row_logprobs)
            else:
                # Fallback to default greedy
                sampled = mx.argmax(row_logprobs, axis=-1)
            
            sampled_tokens.append(sampled)

        # Concatenate along axis 0 to form a (batch_size,) array
        return mx.concatenate(sampled_tokens, axis=0)


class RequestContext:
    """
    Holds the execution state of an active client inference request.
    """
    def __init__(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        response_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        api_key: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.response_queue = response_queue
        self.loop = loop
        self.api_key = api_key
        self.messages = messages
        
        # State variables
        self.cancelled: bool = False
        self.uid: Optional[int] = None
        self.detokenizer: Any = None
        
        # Timing and token count trackers
        self.time_queued: float = time.perf_counter()
        self.time_first_token: Optional[float] = None
        self.completion_tokens: int = 0
        self.cached_tokens: int = 0
        self.generated_text: str = ""


class InferenceScheduler:
    """
    Manages queueing, dynamic batching scheduling, and background thread execution of MLX inference.
    """
    def __init__(self) -> None:
        self.incoming_queue: Queue = Queue()
        self.active_requests: Dict[int, RequestContext] = {}
        self.batch_generator: Optional[BatchGenerator] = None
        # NOTE: LRUPromptCache (mlx_lm non-batching cache) is intentionally NOT
        # used here.  The cache objects it stores are BatchRotatingKVCache instances
        # which are incompatible with BatchGenerator.insert()'s expected per-layer
        # RotatingKVCache format — injecting them causes broadcast-shape crashes.
        # The BatchGenerator manages its own internal caching for active sequences.
        
        self.worker_thread: Optional[Thread] = None
        self.running: bool = False

    def add_request(self, request: RequestContext) -> None:
        """
        Pushes a new request context to the batching queue.
        """
        self.incoming_queue.put(request)
        if settings.metrics_enabled:
            pm.QUEUE_LENGTH.set(self.incoming_queue.qsize())
        logger.info(
            "Enqueued request %s. Queue size: %d",
            request.request_id,
            self.incoming_queue.qsize()
        )

    def start(self) -> None:
        """
        Starts the background worker thread.
        """
        if self.running:
            return
        self.running = True
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Inference worker thread started.")

    def stop(self) -> None:
        """
        Signals the background worker thread to stop and returns immediately.

        We intentionally do NOT join() the thread or call batch_generator.close():
        - The worker thread is daemon=True, so the OS will reap it when the
          process exits anyway — no explicit join is needed.
        - batch_generator.close() and mx.clear_cache() can both block
          indefinitely if there are pending MLX / Metal stream operations.
          Calling either from the shutdown path reliably hangs uvicorn's
          "Waiting for application shutdown." phase.
        - Posting a sentinel None to the queue ensures the thread wakes up
          from its idle queue.get() immediately instead of waiting 100 ms.
        """
        logger.info("[scheduler.stop] signalling worker to stop...")
        self.running = False

        # Wake the thread immediately if it is blocked on queue.get()
        self.incoming_queue.put(None)

        logger.info("[scheduler.stop] signal sent — worker thread will be reaped by OS on exit.")


    def _worker_loop(self) -> None:
        """
        Main worker execution loop running in a dedicated thread.
        """
        # Ensure model is loaded in this thread context
        if model_wrapper.model is None or model_wrapper.tokenizer is None:
            logger.error("Model is not loaded! Cannot start inference loop.")
            self.running = False
            return

        # Import and bind to the unified mlx_lm generation stream
        from mlx_lm.generate import generation_stream
        mx.set_default_stream(generation_stream)

        tokenizer = model_wrapper.tokenizer
        model = model_wrapper.model

        # Initialize the BatchGenerator with continuous batching support and our custom dynamic sampler
        self.batch_generator = BatchGenerator(
            model,
            stop_tokens=set(tokenizer.eos_token_ids),
            sampler=DynamicSampler(self),
            completion_batch_size=settings.max_batch_size,
            prefill_batch_size=settings.max_batch_size
        )

        logger.info("Batch generator initialized and ready.")

        while self.running:
            # 1. Handle dynamic batching & queuing
            # We need to make sure we don't hold the lock or block too long
            requests_to_insert: List[RequestContext] = []
            
            if not self.active_requests:
                # Idle state: block on queue until a request arrives to save CPU.
                # A sentinel None is posted by stop() to unblock this immediately.
                try:
                    req = self.incoming_queue.get(timeout=0.1)
                    if req is None:
                        # Shutdown sentinel — drain task_done and exit loop
                        self.incoming_queue.task_done()
                        break
                    if req.cancelled:
                        self.incoming_queue.task_done()
                        continue
                    requests_to_insert.append(req)
                    self.incoming_queue.task_done()
                except Empty:
                    # Update queue size metrics
                    if settings.metrics_enabled:
                        pm.QUEUE_LENGTH.set(self.incoming_queue.qsize())
                    continue

                # Once the first request arrives, wait up to batching_window_ms to gather more
                start_wait = time.perf_counter()
                wait_duration = settings.batching_window_ms / 1000.0
                while len(requests_to_insert) < settings.max_batch_size:
                    elapsed = time.perf_counter() - start_wait
                    remaining = wait_duration - elapsed
                    if remaining <= 0:
                        break
                    try:
                        req = self.incoming_queue.get(timeout=remaining)
                        if req.cancelled:
                            self.incoming_queue.task_done()
                            continue
                        requests_to_insert.append(req)
                        self.incoming_queue.task_done()
                    except Empty:
                        break
            else:
                # Active generation state: check queue immediately for continuous batch additions
                space_left = settings.max_batch_size - len(self.active_requests)
                while space_left > 0:
                    try:
                        req = self.incoming_queue.get_nowait()
                        if req.cancelled:
                            self.incoming_queue.task_done()
                            continue
                        requests_to_insert.append(req)
                        space_left -= 1
                        self.incoming_queue.task_done()
                    except Empty:
                        break

            # 2. Manage cancellations on active batch
            cancelled_uids = [
                uid for uid, req in self.active_requests.items() if req.cancelled
            ]
            if cancelled_uids:
                logger.info("Removing cancelled requests: %s", cancelled_uids)
                self.batch_generator.remove(cancelled_uids)
                for uid in cancelled_uids:
                    req_ctx = self.active_requests.pop(uid, None)
                    if req_ctx and req_ctx.api_key:
                        key_manager.record_usage(
                            req_ctx.api_key,
                            prompt_tokens=len(req_ctx.prompt_tokens),
                            completion_tokens=req_ctx.completion_tokens,
                            cached_tokens=req_ctx.cached_tokens
                        )

            # Update queue size metrics
            if settings.metrics_enabled:
                pm.QUEUE_LENGTH.set(self.incoming_queue.qsize())

            # 3. Insert newly collected requests into the BatchGenerator
            if requests_to_insert:
                prompts = [r.prompt_tokens for r in requests_to_insert]
                max_tokens = [r.max_tokens for r in requests_to_insert]

                # Always pass caches=None.  LRUPromptCache stores BatchRotatingKVCache
                # objects which are incompatible with the per-layer RotatingKVCache
                # format that BatchGenerator.insert() expects.  Injecting them causes
                # BatchRotatingKVCache.merge() to crash with broadcast-shape errors
                # even for single-request batches.  The BatchGenerator handles its
                # own efficient internal caching for continuations.
                caches = [None] * len(prompts)

                # Insert segments and get assigned internal UIDs
                new_uids = self.batch_generator.insert(
                    prompts=prompts,
                    max_tokens=max_tokens,
                    caches=caches
                )
                # Initialize detokenizer for each request and track them
                for uid, r in zip(new_uids, requests_to_insert):
                    r.uid = uid
                    r.detokenizer = tokenizer.detokenizer
                    r.detokenizer.reset()
                    self.active_requests[uid] = r
                    
                    if settings.metrics_enabled:
                        pm.PROMPT_TOKENS_TOTAL.inc(len(r.prompt_tokens))
                        
                logger.info(
                    "Inserted %d requests into scheduler. Total active: %d",
                    len(requests_to_insert),
                    len(self.active_requests)
                )

            # Update metrics
            if settings.metrics_enabled:
                pm.ACTIVE_GENERATIONS.set(len(self.active_requests))
                pm.BATCH_SIZE.observe(len(self.active_requests))

            # 4. Perform one generation step
            if self.active_requests:
                try:
                    # Run one step of the MLX generation loop
                    responses = self.batch_generator.next()
                except Exception as e:
                    logger.error("Inference step crashed: %s", str(e), exc_info=True)
                    # Notify all active requests of the error and clear active dict
                    for uid, r in list(self.active_requests.items()):
                        r.loop.call_soon_threadsafe(
                            r.response_queue.put_nowait,
                            (None, "error")
                        )
                    self.active_requests.clear()
                    
                    # Reset BatchGenerator to clear any corrupted states
                    if self.batch_generator is not None:
                        try:
                            self.batch_generator.close()
                        except Exception:
                            pass
                        self.batch_generator = None
                    try:
                        self.batch_generator = BatchGenerator(
                            model,
                            stop_tokens=set(tokenizer.eos_token_ids),
                            sampler=DynamicSampler(self),
                            completion_batch_size=settings.max_batch_size,
                            prefill_batch_size=settings.max_batch_size
                        )
                    except Exception as reinit_err:
                        logger.critical("Failed to reinitialize BatchGenerator: %s", str(reinit_err))
                    continue

                # Process results
                for r_gen in responses:
                    req_ctx = self.active_requests.get(r_gen.uid)
                    if not req_ctx:
                        continue

                    # Record Time To First Token (TTFT)
                    if req_ctx.time_first_token is None:
                        req_ctx.time_first_token = time.perf_counter()
                        ttft = req_ctx.time_first_token - req_ctx.time_queued
                        if settings.metrics_enabled:
                            pm.TIME_TO_FIRST_TOKEN_SECONDS.observe(ttft)
                        logger.info("TTFT for request %s: %.3f sec", req_ctx.request_id, ttft)

                    # Feed the token into the streaming detokenizer
                    req_ctx.detokenizer.add_token(r_gen.token)
                    req_ctx.completion_tokens += 1
                    text_segment = req_ctx.detokenizer.last_segment
                    if text_segment:
                        req_ctx.generated_text += text_segment

                    # Send generated text token segment to client
                    req_ctx.loop.call_soon_threadsafe(
                        req_ctx.response_queue.put_nowait,
                        (text_segment, None)
                    )

                    # Check if sequence is finished
                    if r_gen.finish_reason is not None:
                        # Finalize detokenizer buffer
                        req_ctx.detokenizer.finalize()
                        final_segment = req_ctx.detokenizer.last_segment
                        if final_segment:
                            req_ctx.generated_text += final_segment
                            req_ctx.loop.call_soon_threadsafe(
                                req_ctx.response_queue.put_nowait,
                                (final_segment, None)
                            )

                        # End-of-Generation stats
                        time_finished = time.perf_counter()
                        total_latency = time_finished - req_ctx.time_queued
                        gen_latency = time_finished - (req_ctx.time_first_token or time_finished)
                        
                        tps = 0.0
                        if gen_latency > 0:
                            tps = req_ctx.completion_tokens / gen_latency

                        if settings.metrics_enabled:
                            pm.GENERATED_TOKENS_TOTAL.inc(req_ctx.completion_tokens)
                            pm.REQUEST_LATENCY_SECONDS.observe(total_latency)
                            if tps > 0:
                                pm.TOKENS_GENERATED_SPEED.observe(tps)

                        # Log the completion details structuredly
                        logger.info(
                            "Request completed",
                            extra={
                                "request_id": req_ctx.request_id,
                                "latency_ms": int(total_latency * 1000),
                                "model": model_wrapper.model_name or "unknown",
                                "prompt_tokens": len(req_ctx.prompt_tokens),
                                "generated_tokens": req_ctx.completion_tokens,
                                "tokens_per_sec": round(tps, 2),
                                "batch_size": len(self.active_requests),
                            }
                        )

                        # Record token usage to KeyManager
                        if req_ctx.api_key:
                            key_manager.record_usage(
                                req_ctx.api_key,
                                prompt_tokens=len(req_ctx.prompt_tokens),
                                completion_tokens=req_ctx.completion_tokens,
                                cached_tokens=req_ctx.cached_tokens
                            )

                        # Send termination sentinel to client queue
                        req_ctx.loop.call_soon_threadsafe(
                            req_ctx.response_queue.put_nowait,
                            (None, r_gen.finish_reason)
                        )
                        
                        # Remove from tracking list
                        self.active_requests.pop(r_gen.uid, None)

            # Avoid tight spin if no work was scheduled in this iteration
            if not self.active_requests:
                time.sleep(0.005)


# Global scheduler instance
inference_scheduler = InferenceScheduler()
