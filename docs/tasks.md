# Tasks: Hailo LLM Server

## Phase 0: Discovery
- [x] Confirm available `.hef` model files on the system and select the target
- [x] Probe `hailo_platform.genai` SDK: document `LLM` class interface (init args, inference call signature, streaming mechanism)
- [x] Run a minimal throwaway inference script to observe raw token output
- [x] Capture and document raw tool-call output from the model (tags, format, common malformations)
- [x] Confirm max sequence length / context window baked into the HEF

> See `docs/discovery.md` for full findings.

## Pre-Phase: Environment Cleanup
- [x] `apt purge hailort hailort-pcie-driver` to remove 5.2.0 config remnants
- [x] Verify `h10-hailort` 5.1.1 is intact and inference still works after purge
- [x] `apt-mark manual h10-hailort h10-hailort-pcie-driver python3-h10-hailort` to prevent accidental autoremove

## Phase 1: Foundation & Inference Core
- [x] Initialize `uv` project with `--system-site-packages` in `src/npu_server`
- [x] Add dependencies: `fastapi`, `uvicorn`, `pydantic`, `jinja2`
- [x] Define config layer: model path, inference defaults, server port, log level (env vars or config file)
- [x] Implement `HailoProvider` class: load `VDevice` + `LLM` from config; expose `generate()` and `generate_all()`
- [x] Add explicit error handling for device faults and inference timeouts in `HailoProvider`
- [x] Write basic inference test script (raw NPU generation, no FastAPI)

## Phase 2: API & Prompt Management
- [x] Create FastAPI application skeleton
- [x] Define Pydantic models for OpenAI `/v1/chat/completions` request and response
- [x] Fetch and cache `llm.prompt_template()` at startup; render via Jinja2 for each request
- [x] Strip stop tokens (`<|im_end|>`, `<|endoftext|>`) from outgoing token stream
- [x] Implement SSE streaming as primary response path; non-streaming wraps the same generator
- [x] Add `asyncio.Lock` concurrency guard — concurrent requests receive `429`
- [x] Track context token usage; reject requests that would overflow 2048 tokens with `400`
- [x] Implement `GET /health` endpoint: NPU device status, context usage, service info

## Phase 3: Productionization
- [x] Add structured logging: TTFT, tokens/sec, context usage, errors per request
- [x] Add periodic background task logging NPU temperature and device status
- [x] Implement inference timeout — abort and return `503` if NPU stalls
- [x] Implement device fault recovery: attempt re-init on `HailoRTStatusException`
- [x] Ensure `asyncio.Lock` is released on all exception paths
- [x] Implement graceful shutdown: `llm.release()` and `vdevice.release()` on SIGTERM
- [x] Create `hailo-llm-server.service` systemd unit file (boot start, restart on failure)
- [x] Write `README.md`: setup, config reference, `curl` examples for streaming and non-streaming

## Phase 4: Usage Monitoring
- [x] Design SQLite schema: timestamp, request_id, prompt_tokens, completion_tokens, ttft_ms, tokens_per_sec, context_usage, model, error
- [x] Implement metrics writer — non-blocking write to SQLite after each completed request
- [x] Implement `GET /stats` endpoint: request count, avg/p95 TTFT, avg tokens/sec, error rate — filterable by time window (1h, 24h, all)
- [x] Add `add dependency: aiosqlite` (async SQLite writes won't block the event loop)
