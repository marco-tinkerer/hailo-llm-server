import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from jinja2 import Environment

from hailo_platform.pyhailort.pyhailort import HailoRTException  # type: ignore[import]

from .config import settings
from .metrics import MetricsDB, RequestRecord
from .models import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaMessage,
    HealthResponse,
    StatsWindow,
    StreamChoice,
    UsageInfo,
)
from .provider import HailoProvider, HailoProviderError

logger = logging.getLogger(__name__)

_provider: HailoProvider | None = None
_jinja_env: Environment | None = None
_prompt_template_str: str | None = None
_inference_lock = asyncio.Lock()
_telemetry_task: asyncio.Task | None = None
_metrics_db: MetricsDB | None = None


async def _telemetry_loop() -> None:
    while True:
        await asyncio.sleep(60)
        if _provider is None:
            continue
        loop = asyncio.get_running_loop()
        temps = await loop.run_in_executor(None, _provider.get_temperature)
        if temps is not None:
            logger.info("npu_telemetry ts0=%.1f°C ts1=%.1f°C", temps[0], temps[1])
        else:
            logger.debug("npu_telemetry temperature unavailable")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _provider, _jinja_env, _prompt_template_str, _telemetry_task, _metrics_db

    _provider = HailoProvider(settings)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _provider.initialize)

    _prompt_template_str = _provider.prompt_template()
    _jinja_env = Environment()
    _jinja_env.filters["tojson"] = json.dumps

    _metrics_db = MetricsDB(settings.metrics_db_path)
    await _metrics_db.setup()

    _telemetry_task = asyncio.create_task(_telemetry_loop())
    logger.info("Server ready on %s:%d", settings.server_host, settings.server_port)

    try:
        yield
    finally:
        if _telemetry_task is not None:
            _telemetry_task.cancel()
        if _metrics_db is not None:
            await _metrics_db.close()
        if _provider is not None:
            await loop.run_in_executor(None, _provider.release)


app = FastAPI(title="NPU LLM Server", version="0.1.0", lifespan=lifespan)


def _get_provider() -> HailoProvider:
    if _provider is None:
        raise HTTPException(status_code=503, detail="NPU not initialized")
    return _provider


def _render_prompt(request: ChatCompletionRequest) -> str:
    assert _jinja_env is not None and _prompt_template_str is not None
    tmpl = _jinja_env.from_string(_prompt_template_str)
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    kwargs: dict = {"messages": messages, "add_generation_prompt": True}
    if request.tools:
        kwargs["tools"] = request.tools
    return tmpl.render(**kwargs)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _check_context(provider: HailoProvider, prompt: str, max_tokens: int) -> None:
    used = provider.get_context_usage()
    estimated_prompt = _estimate_tokens(prompt)
    if used + estimated_prompt + max_tokens > provider.max_context_capacity():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "context_overflow",
                "context_used": used,
                "estimated_prompt_tokens": estimated_prompt,
                "max_generated_tokens": max_tokens,
                "context_max": provider.max_context_capacity(),
            },
        )


async def _try_recover(provider: HailoProvider) -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, provider.reinitialize)
        logger.info("NPU recovery succeeded")
    except Exception:
        logger.error("NPU recovery failed", exc_info=True)


async def _record(
    request_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: float,
    tokens_per_sec: float,
    context_usage: int,
    error: str | None = None,
) -> None:
    if _metrics_db is None:
        return
    await _metrics_db.record(
        RequestRecord(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tokens_per_sec=tokens_per_sec,
            context_usage=context_usage,
            model=model,
            error=error,
        )
    )


async def _stream_response(
    request: ChatCompletionRequest,
    provider: HailoProvider,
    prompt: str,
    max_tokens: int,
    request_id: str,
    request_start: float,
) -> AsyncIterator[str]:
    if _inference_lock.locked():
        err = {"error": {"message": "NPU busy", "type": "rate_limit_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    async with _inference_lock:
        first_chunk = ChatCompletionChunk(
            id=request_id,
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"), finish_reason=None)],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        ttft_s: float | None = None
        error_str: str | None = None
        metrics = None

        try:
            async with provider.generate(prompt, max_tokens) as (stream, metrics):
                async for token in stream:
                    if ttft_s is None:
                        ttft_s = time.monotonic() - request_start
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        model=request.model,
                        choices=[StreamChoice(delta=DeltaMessage(content=token), finish_reason=None)],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

        except HailoRTException as exc:
            error_str = f"device_fault: {exc}"
            logger.error("request_id=%s NPU device fault: %s — attempting recovery", request_id, exc)
            await _try_recover(provider)
            yield "data: [DONE]\n\n"
        except HailoProviderError as exc:
            error_str = str(exc)
            logger.error("request_id=%s inference error: %s", request_id, exc)
            yield "data: [DONE]\n\n"
        else:
            final_chunk = ChatCompletionChunk(
                id=request_id,
                model=request.model,
                choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        ctx = provider.get_context_usage()
        m_ttft = ttft_s or 0.0
        m_toks = metrics.completion_tokens if metrics else 0
        m_tps = metrics.toks_per_sec if metrics else 0.0

        logger.info(
            "request_id=%s ttft=%.3fs tok/s=%.1f tokens=%d context=%d/%d%s",
            request_id, m_ttft, m_tps, m_toks, ctx,
            provider.max_context_capacity(),
            f" error={error_str}" if error_str else "",
        )
        await _record(
            request_id=request_id,
            model=request.model,
            prompt_tokens=ctx,
            completion_tokens=m_toks,
            ttft_ms=m_ttft * 1000,
            tokens_per_sec=m_tps,
            context_usage=ctx,
            error=error_str,
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    provider = _get_provider()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    request_start = time.monotonic()

    prompt = _render_prompt(request)
    max_tokens = request.max_tokens or settings.max_generated_tokens
    _check_context(provider, prompt, max_tokens)

    if request.stream:
        return StreamingResponse(
            _stream_response(request, provider, prompt, max_tokens, request_id, request_start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _inference_lock.locked():
        raise HTTPException(status_code=429, detail="NPU busy — only one concurrent request allowed")

    error_str: str | None = None
    async with _inference_lock:
        try:
            text, metrics = await provider.generate_all(prompt, max_tokens)
        except HailoRTException as exc:
            error_str = f"device_fault: {exc}"
            logger.error("request_id=%s NPU device fault: %s — attempting recovery", request_id, exc)
            await _try_recover(provider)
            await _record(request_id, request.model, 0, 0, 0.0, 0.0, 0, error=error_str)
            raise HTTPException(status_code=503, detail="NPU device fault; recovery attempted — retry request")
        except HailoProviderError as exc:
            error_str = str(exc)
            await _record(request_id, request.model, 0, 0, 0.0, 0.0, 0, error=error_str)
            raise HTTPException(status_code=503, detail=error_str)

    ctx = provider.get_context_usage()
    logger.info(
        "request_id=%s ttft=%.3fs tok/s=%.1f tokens=%d context=%d/%d",
        request_id, metrics.ttft_s, metrics.toks_per_sec,
        metrics.completion_tokens, ctx, provider.max_context_capacity(),
    )
    await _record(
        request_id=request_id,
        model=request.model,
        prompt_tokens=ctx,
        completion_tokens=metrics.completion_tokens,
        ttft_ms=metrics.ttft_s * 1000,
        tokens_per_sec=metrics.toks_per_sec,
        context_usage=ctx,
    )

    return ChatCompletionResponse(
        id=request_id,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=ctx,
            completion_tokens=metrics.completion_tokens,
            total_tokens=ctx + metrics.completion_tokens,
        ),
    )


@app.post("/v1/context/clear", status_code=204)
async def clear_context():
    provider = _get_provider()
    if _inference_lock.locked():
        raise HTTPException(status_code=429, detail="NPU busy")
    async with _inference_lock:
        provider.clear_context()
    logger.info("Context cleared")


@app.get("/health", response_model=HealthResponse)
async def health():
    provider = _get_provider()
    used = provider.get_context_usage()
    cap = provider.max_context_capacity()
    return HealthResponse(
        status="ok",
        context_used=used,
        context_max=cap,
        context_free=cap - used,
        model=str(settings.model_path.name),
        lora=settings.lora_name,
    )


@app.get("/stats", response_model=StatsWindow)
async def stats(
    window: str = Query(default="24h", description="Time window: 1h, 24h, or all"),
):
    if _metrics_db is None:
        raise HTTPException(status_code=503, detail="Metrics DB not initialized")

    window_hours: float | None
    if window == "all":
        window_hours = None
    elif window.endswith("h"):
        try:
            window_hours = float(window[:-1])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid window — use 1h, 24h, or all")
    else:
        raise HTTPException(status_code=400, detail="Invalid window — use 1h, 24h, or all")

    data = await _metrics_db.stats(window_hours)
    return StatsWindow(**data)
