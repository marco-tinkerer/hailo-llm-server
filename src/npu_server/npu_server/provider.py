import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from hailo_platform.genai import LLM, VDevice  # type: ignore[import]
from hailo_platform.pyhailort.pyhailort import HailoRTException  # type: ignore[import]

from .config import Settings

logger = logging.getLogger(__name__)

STOP_TOKENS = frozenset(["<|im_end|>", "<|endoftext|>"])


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

    def initialize(self) -> None:
        logger.info("Initializing Hailo NPU device")
        self._vdevice = VDevice()
        kwargs: dict = {"lora_name": self._settings.lora_name} if self._settings.lora_name else {}
        self._llm = LLM(self._vdevice, str(self._settings.model_path), **kwargs)
        logger.info("NPU device ready, max context: %d tokens", self._llm.max_context_capacity())

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
    ) -> AsyncIterator[tuple[AsyncIterator[str], GenerateMetrics]]:
        """Async streaming generator. Yields (token_stream, metrics).
        Metrics are populated after the stream is exhausted."""
        llm = self._require_llm()
        max_tokens = max_generated_tokens or self._settings.max_generated_tokens
        timeout = self._settings.inference_timeout_s
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
                    max_generated_tokens=max_tokens,
                    do_sample=self._settings.do_sample,
                ) as gen:
                    for token in gen:
                        if token in STOP_TOKENS:
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

    async def generate_all(self, prompt: str, max_generated_tokens: int | None = None) -> tuple[str, GenerateMetrics]:
        parts: list[str] = []
        async with self.generate(prompt, max_generated_tokens) as (stream, metrics):
            async for token in stream:
                parts.append(token)
        return "".join(parts), metrics
