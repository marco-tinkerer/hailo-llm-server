import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from pathlib import Path

from hailo_platform.genai import LLM, VDevice  # type: ignore[import]
from hailo_platform.pyhailort.pyhailort import HailoRTException  # type: ignore[import]

from .config import Settings

logger = logging.getLogger(__name__)

DEFAULT_STOP_TOKENS = frozenset(
    [
        "<|im_end|>",
        "<|endoftext|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|eom_id|>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
    ]
)


class HailoProviderError(RuntimeError):
    pass


@dataclass
class GenerateMetrics:
    ttft_s: float = 0.0
    completion_tokens: int = 0
    elapsed_s: float = 0.0
    toks_per_sec: float = 0.0
    error: str | None = None


class HailoProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._vdevice: VDevice | None = None
        self._llm: LLM | None = None
        self._stop_tokens: set[str] = set(DEFAULT_STOP_TOKENS)

    @property
    def model_path(self) -> Path:
        return self._settings.model_path

    @property
    def model_name(self) -> str:
        return self._settings.model_path.name

    @property
    def lora_name(self) -> str:
        return self._settings.lora_name

    def initialize(self) -> None:
        logger.info("Initializing Hailo NPU device with model: %s", self._settings.model_path)
        self._vdevice = VDevice()
        kwargs: dict = {"lora_name": self._settings.lora_name} if self._settings.lora_name else {}
        self._llm = LLM(self._vdevice, str(self._settings.model_path), **kwargs)
        self._stop_tokens = set(DEFAULT_STOP_TOKENS)
        try:
            model_stops = self._llm.get_stop_tokens()
            if model_stops:
                self._stop_tokens.update(model_stops)
        except Exception:
            logger.debug("Could not query model stop tokens", exc_info=True)
        logger.info(
            "NPU device ready (model=%s, max_context=%d, stop_tokens=%s)",
            self.model_name,
            self._llm.max_context_capacity(),
            sorted(self._stop_tokens),
        )

    def load_model(self, model_path: Path, lora_name: str = "") -> None:
        logger.info("Switching NPU model to %s (lora=%r)", model_path, lora_name)
        self.release()
        self._settings.model_path = model_path
        self._settings.lora_name = lora_name
        self.initialize()

    def reinitialize(self) -> None:
        logger.warning("Re-initializing NPU device after fault")
        self.release()
        self.initialize()

    def release(self) -> None:
        if self._llm is not None:
            try:
                self._llm.release()
            except Exception:
                logger.warning("Error releasing LLM", exc_info=True)
            self._llm = None
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                logger.warning("Error releasing VDevice", exc_info=True)
            self._vdevice = None
        logger.info("NPU device released")

    def _require_llm(self) -> LLM:
        if self._llm is None:
            raise HailoProviderError("NPU not initialized")
        return self._llm

    def prompt_template(self) -> str:
        return self._require_llm().prompt_template()

    def count_tokens(self, text: str) -> int:
        try:
            return len(self._require_llm().tokenize(text))
        except Exception:
            return max(1, len(text) // 4)

    def max_context_capacity(self) -> int:
        return self._require_llm().max_context_capacity()

    def get_context_usage(self) -> int:
        return self._require_llm().get_context_usage_size()

    def clear_context(self) -> None:
        self._require_llm().clear_context()

    def get_temperature(self) -> tuple[float, float] | None:
        """Returns (ts0, ts1) chip temperatures in °C, or None on failure."""
        try:
            from hailo_platform import Device  # type: ignore[import]
            d = Device()
            try:
                temp = d.control.get_chip_temperature()
                return (temp.ts0_temperature, temp.ts1_temperature)
            finally:
                d.release()
        except Exception as exc:
            logger.debug("Could not read chip temperature: %s", exc)
            return None

    @asynccontextmanager
    async def generate(
        self,
        prompt: str,
        max_generated_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[tuple[AsyncIterator[str], GenerateMetrics]]:
        """Async streaming generator. Yields (token_stream, metrics).
        Metrics are populated after the stream is exhausted."""
        llm = self._require_llm()
        max_tokens = max_generated_tokens or self._settings.max_generated_tokens
        timeout = self._settings.inference_timeout_s

        eff_temp = temperature if temperature is not None else self._settings.temperature
        eff_freq = frequency_penalty if frequency_penalty is not None else self._settings.frequency_penalty
        if eff_temp > 0.0:
            do_sample = True
            gen_temp: float | None = eff_temp
            gen_top_p = top_p if top_p is not None else self._settings.top_p
            gen_top_k = top_k if top_k is not None else self._settings.top_k
            gen_seed = seed
        else:
            do_sample = self._settings.do_sample
            gen_temp = None
            gen_top_p = None
            gen_top_k = None
            gen_seed = None

        stop_tokens = self._stop_tokens
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        error_holder: list[BaseException] = []
        metrics = GenerateMetrics()

        def _run_inference() -> None:
            t_start = time.monotonic()
            first_token = True
            try:
                with llm.generate(
                    prompt,
                    temperature=gen_temp,
                    top_p=gen_top_p,
                    top_k=gen_top_k,
                    frequency_penalty=eff_freq,
                    max_generated_tokens=max_tokens,
                    do_sample=do_sample,
                    seed=gen_seed,
                ) as gen:
                    for token in gen:
                        if token in stop_tokens:
                            continue
                        if first_token:
                            metrics.ttft_s = time.monotonic() - t_start
                            first_token = False
                        metrics.completion_tokens += 1
                        loop.call_soon_threadsafe(token_queue.put_nowait, token)
            except HailoRTException as exc:
                error_holder.append(exc)
            except Exception as exc:
                error_holder.append(exc)
            finally:
                metrics.elapsed_s = time.monotonic() - t_start
                if metrics.elapsed_s > 0 and metrics.completion_tokens > 0:
                    metrics.toks_per_sec = metrics.completion_tokens / metrics.elapsed_s
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        loop.run_in_executor(None, _run_inference)

        async def _stream() -> AsyncIterator[str]:
            while True:
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    metrics.error = f"NPU inference stalled: no token within {timeout}s"
                    raise HailoProviderError(metrics.error)
                if token is None:
                    break
                yield token
            if error_holder:
                exc = error_holder[0]
                metrics.error = str(exc)
                # Surface HailoRTException so caller can attempt recovery
                if isinstance(exc, HailoRTException):
                    raise exc
                raise HailoProviderError(f"NPU inference failed: {exc}") from exc

        yield _stream(), metrics

    async def generate_all(
        self,
        prompt: str,
        max_generated_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
    ) -> tuple[str, GenerateMetrics]:
        parts: list[str] = []
        async with self.generate(
            prompt,
            max_generated_tokens=max_generated_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            seed=seed,
        ) as (stream, metrics):
            async for token in stream:
                parts.append(token)
        return "".join(parts), metrics
