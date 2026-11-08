import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path
from typing import Any

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
    LoadModelRequest,
    ModelCard,
    ModelListResponse,
    StatsWindow,
    StreamChoice,
    UsageInfo,
)
from .provider import HailoProvider, HailoProviderError

logger = logging.getLogger(__name__)

_provider: HailoProvider | None = None
_jinja_env: Environment | None = None
_inference_lock = asyncio.Lock()
_telemetry_task: asyncio.Task | None = None
_metrics_db: MetricsDB | None = None


def _json_dumps_filter(val: Any, indent: int | None = None) -> str:
    return json.dumps(val, indent=indent)


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
    global _provider, _jinja_env, _telemetry_task, _metrics_db

    _provider = HailoProvider(settings)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _provider.initialize)

    _jinja_env = Environment()
    _jinja_env.filters["tojson"] = _json_dumps_filter
    _jinja_env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)

    _metrics_db = MetricsDB(settings.metrics_db_path)
    await _metrics_db.setup()

    _telemetry_task = asyncio.create_task(_telemetry_loop())
    logger.info("Server ready on %s:%d (model=%s)", settings.server_host, settings.server_port, _provider.model_name)

    try:
        yield
    finally:
        if _telemetry_task is not None:
            _telemetry_task.cancel()
        if _metrics_db is not None:
            await _metrics_db.close()
        if _provider is not None:
            await loop.run_in_executor(None, _provider.release)


app = FastAPI(title="NPU LLM Server", version="0.2.0", lifespan=lifespan)


def _get_provider() -> HailoProvider:
    if _provider is None:
        raise HTTPException(status_code=503, detail="NPU not initialized")
    return _provider


def _resolve_model_path(model_str: str) -> Path | None:
    if not model_str:
        return None
    models_dir = settings.model_path.parent
    candidate_name = Path(model_str).name
    if not candidate_name.endswith(".hef"):
        candidate_name = f"{candidate_name}.hef"
    candidate = models_dir / candidate_name
    if candidate.is_file():
        return candidate
    direct = Path(model_str)
    if direct.is_file() and direct.suffix == ".hef":
        return direct
    return None


async def _ensure_model_loaded(provider: HailoProvider, requested_model: str) -> None:
    target = _resolve_model_path(requested_model)
    if target is not None and target.resolve() != provider.model_path.resolve():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, provider.load_model, target, "")


def _render_prompt(request: ChatCompletionRequest, provider: HailoProvider) -> str:
    assert _jinja_env is not None
    template_str = provider.prompt_template()
    tmpl = _jinja_env.from_string(template_str)
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    bos_token = "<|begin_of_text|>" if "llama" in provider.model_name.lower() else ""
    kwargs: dict[str, Any] = {
        "messages": messages,
        "add_generation_prompt": True,
        "bos_token": bos_token,
    }
    if request.tools:
        kwargs["tools"] = request.tools
    return tmpl.render(**kwargs)


def _prepare_context_and_budget(
    provider: HailoProvider,
    prompt: str,
    requested_max_tokens: int,
    preserve_context: bool,
) -> tuple[int, int]:
    """Clears context if stateless, counts prompt tokens, and returns (prompt_tokens, clamped_max_tokens)."""
    if not preserve_context:
        provider.clear_context()
    used = provider.get_context_usage()
    prompt_tokens = provider.count_tokens(prompt)
    cap = provider.max_context_capacity()
    available = cap - used - prompt_tokens
    if available <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "context_overflow",
                "context_used": used,
                "prompt_tokens": prompt_tokens,
                "requested_max_tokens": requested_max_tokens,
                "context_max": cap,
            },
        )
    effective_max_tokens = min(requested_max_tokens, available)
    return prompt_tokens, effective_max_tokens


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
        try:
            await _ensure_model_loaded(provider, request.model)
            prompt = _render_prompt(request, provider)
            prompt_tokens, effective_max_tokens = _prepare_context_and_budget(
                provider, prompt, max_tokens, request.preserve_context
            )
        except HTTPException as exc:
            err = {"error": {"message": str(exc.detail), "type": "invalid_request_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            yield "data: [DONE]\n\n"
            return

        active_model = provider.model_name
        first_chunk = ChatCompletionChunk(
            id=request_id,
            model=active_model,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"), finish_reason=None)],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        ttft_s: float | None = None
        error_str: str | None = None
        metrics = None

        try:
            async with provider.generate(
                prompt,
                max_generated_tokens=effective_max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                seed=request.seed,
            ) as (stream, metrics):
                async for token in stream:
                    if ttft_s is None:
                        ttft_s = time.monotonic() - request_start
                    chunk = ChatCompletionChunk(
                        id=request_id,
                        model=active_model,
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
                model=active_model,
                choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        ctx = provider.get_context_usage()
        m_ttft = ttft_s or 0.0
        m_toks = metrics.completion_tokens if metrics else 0
        m_tps = metrics.toks_per_sec if metrics else 0.0

        logger.info(
            "request_id=%s model=%s ttft=%.3fs tok/s=%.1f prompt=%d tokens=%d context=%d/%d%s",
            request_id, active_model, m_ttft, m_tps, prompt_tokens, m_toks, ctx,
            provider.max_context_capacity(),
            f" error={error_str}" if error_str else "",
        )
        await _record(
            request_id=request_id,
            model=active_model,
            prompt_tokens=prompt_tokens,
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
    max_tokens = request.max_tokens or settings.max_generated_tokens

    if request.stream:
        return StreamingResponse(
            _stream_response(request, provider, max_tokens, request_id, request_start),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _inference_lock.locked():
        raise HTTPException(status_code=429, detail="NPU busy — only one concurrent request allowed")

    error_str: str | None = None
    async with _inference_lock:
        await _ensure_model_loaded(provider, request.model)
        active_model = provider.model_name
        prompt = _render_prompt(request, provider)
        prompt_tokens, effective_max_tokens = _prepare_context_and_budget(
            provider, prompt, max_tokens, request.preserve_context
        )

        try:
            text, metrics = await provider.generate_all(
                prompt,
                max_generated_tokens=effective_max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                seed=request.seed,
            )
        except HailoRTException as exc:
            error_str = f"device_fault: {exc}"
            logger.error("request_id=%s NPU device fault: %s — attempting recovery", request_id, exc)
            await _try_recover(provider)
            await _record(request_id, active_model, prompt_tokens, 0, 0.0, 0.0, 0, error=error_str)
            raise HTTPException(status_code=503, detail="NPU device fault; recovery attempted — retry request")
        except HailoProviderError as exc:
            error_str = str(exc)
            await _record(request_id, active_model, prompt_tokens, 0, 0.0, 0.0, 0, error=error_str)
            raise HTTPException(status_code=503, detail=error_str)

        ctx = provider.get_context_usage()

    logger.info(
        "request_id=%s model=%s ttft=%.3fs tok/s=%.1f prompt=%d tokens=%d context=%d/%d",
        request_id, active_model, metrics.ttft_s, metrics.toks_per_sec,
        prompt_tokens, metrics.completion_tokens, ctx, provider.max_context_capacity(),
    )
    await _record(
        request_id=request_id,
        model=active_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        ttft_ms=metrics.ttft_s * 1000,
        tokens_per_sec=metrics.toks_per_sec,
        context_usage=ctx,
    )

    return ChatCompletionResponse(
        id=request_id,
        model=active_model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            total_tokens=prompt_tokens + metrics.completion_tokens,
        ),
    )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    provider = _get_provider()
    models_dir = settings.model_path.parent
    cards: list[ModelCard] = []
    if models_dir.is_dir():
        for hef in sorted(models_dir.glob("*.hef")):
            cards.append(
                ModelCard(
                    id=hef.name,
                    created=int(hef.stat().st_mtime),
                    active=(hef.name == provider.model_name),
                )
            )
    return ModelListResponse(data=cards)


@app.post("/v1/models/load", response_model=HealthResponse)
async def load_model(req: LoadModelRequest):
    provider = _get_provider()
    target = _resolve_model_path(req.model)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {req.model}")
    if _inference_lock.locked():
        raise HTTPException(status_code=429, detail="NPU busy")
    async with _inference_lock:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, provider.load_model, target, req.lora_name)
        used = provider.get_context_usage()
        cap = provider.max_context_capacity()
    return HealthResponse(
        status="ok",
        context_used=used,
        context_max=cap,
        context_free=cap - used,
        model=provider.model_name,
        lora=provider.lora_name,
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
        model=provider.model_name,
        lora=provider.lora_name,
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
