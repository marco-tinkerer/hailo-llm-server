# Implementation Plan: Hailo LLM Server

## Pre-Phase: Environment Cleanup
### Goal: Ensure a clean, unambiguous system state before writing any code.

1. **Purge hailort 5.2.0 remnants:**
   - `hailort` and `hailort-pcie-driver` 5.2.0 are in a partially-removed (`ic`) state, leaving config files that may conflict with the active `h10-hailort` 5.1.1 packages.
   - Run `apt purge hailort hailort-pcie-driver` to remove config remnants.
   - Verify `h10-hailort` 5.1.1 remains intact and inference still works after purge.

---

## Phase 1: Foundation & Inference Core
### Goal: Establish a clean environment and execute reliable raw inference via the NPU.

1.  **Environment Setup:**
    - Initialize `uv` project with `--system-site-packages` in `src/npu_server`.
    - Dependencies: `fastapi`, `uvicorn`, `pydantic`, `jinja2`.
2.  **Configuration:**
    - Define a config layer (environment variables or TOML/YAML file) for: model path, inference defaults (temperature, max tokens), server port, log level.
    - No hardcoded paths or parameters anywhere in the codebase.
3.  **Hardware Binding:**
    - Implement the `HailoProvider` class.
    - Initialize `VDevice` and `LLM` from `hailo_platform.genai`. Model path loaded from config.
    - Expose `generate()` (streaming) and `generate_all()` (blocking) methods.
    - Handle device errors and inference timeouts explicitly — no unhandled exceptions.
4.  **Basic Inference Test:**
    - Write a throwaway test script confirming the NPU generates text correctly outside FastAPI.

---

## Phase 2: The API
### Goal: Expose the NPU as a reliable, OpenAI-compatible network service.

1.  **API Structure:**
    - Create the FastAPI application.
    - Define Pydantic models for the OpenAI `/v1/chat/completions` request and response format.
2.  **Prompt Templating:**
    - Fetch the model's Jinja2 template via `llm.prompt_template()` at startup.
    - Render incoming `messages` arrays into the model's required format (ChatML).
    - Strip stop tokens (`<|im_end|>`, `<|endoftext|>`) from the outgoing token stream.
3.  **Streaming Architecture:**
    - Implement SSE streaming as the primary response path from the start.
    - Non-streaming (`stream=false`) wraps the same internal generator — no separate code path.
4.  **Concurrency & Context Management:**
    - `asyncio.Lock` ensures single-tenant NPU access. Concurrent requests receive `429`.
    - Track context token usage per request. Reject requests that would overflow 2048 tokens with a clear `400` error before inference begins.
    - Expose `clear_context()` control so callers can explicitly reset conversation state.
5.  **Health & Telemetry Endpoint:**
    - `GET /health` returns NPU device status, current context usage, and basic service info.
    - Lays the foundation for Phase 3 telemetry without blocking API development.

---

## Phase 3: Productionization
### Goal: Make the service resilient, observable, and deployable.

1.  **Structured Logging & Metrics:**
    - Log every request: TTFT, tokens/sec, context usage, model used, errors.
    - Periodic background task to log NPU hardware telemetry (temperature, device status) via `get_physical_devices()`.
2.  **Resilience:**
    - Inference timeout: if the NPU does not produce a token within a configurable window, abort and return `503`.
    - Device fault recovery: on `HailoRTStatusException`, attempt re-initialization before failing the request.
    - Ensure the `asyncio.Lock` is always released even on exception paths.
3.  **System Integration:**
    - Create `hailo-llm-server.service` systemd unit file.
    - Configure for boot start and automatic restart on failure.
    - Ensure graceful shutdown releases the NPU (`llm.release()`, `vdevice.release()`).
4.  **Documentation:**
    - `README.md` with setup instructions, configuration reference, and `curl` examples for streaming and non-streaming requests.
