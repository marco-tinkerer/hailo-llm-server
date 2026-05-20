# Requirements: Hailo LLM Server

## 1. Project Vision
A rock-solid, universal LLM inference primitive designed specifically for the Raspberry Pi 5 AI HAT+ (Hailo-10H NPU). This service runs as a standalone background process, exposing the NPU's capabilities via a clean, reliable API. It provides pure language inference — reasoning and generation — and nothing else. Tool orchestration and agentic workflows are the responsibility of calling services.

## 2. System Architecture

This server is one layer in a three-layer stack:

```
┌─────────────────────────────────────────────────────┐
│              Client Applications                    │
│   (home automation, tactical mesh, local tools)     │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────┐
│          LangGraph Agent Service                    │
│  (per-project, handles orchestration & tool calls)  │
│  - Tactical Mesh Pi: co-located on that Pi          │
│  - Other projects: deployed where appropriate       │
└────────────────────┬────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────┐
│           Hailo LLM Server  ← THIS SERVICE          │
│     (pure inference, no tool or project logic)      │
└─────────────────────────────────────────────────────┘
```

The LLM server has no knowledge of tools, workflows, or any project-specific logic. Keeping it a clean primitive ensures it can serve any future project without modification.

## 3. Infrastructure & Environment
- **Hardware:** Raspberry Pi 5 (8GB) + AI HAT+ 2 (Hailo-10H).
- **Hailo SDK:** `h10-hailort` 5.1.1 (system-level) with `hailo_platform` Python bindings.
- **Python Management:** `uv` using `--system-site-packages` to bridge system Hailo bindings into the virtual environment.
- **Framework:** FastAPI / Uvicorn for asynchronous REST serving.
- **Model:** `Qwen2.5-1.5B-Instruct.hef` — the base chat model. Loaded path and parameters are configuration-driven, not hardcoded.

## 4. Core API Requirements

- **Interface:** OpenAI-compatible `/v1/chat/completions` endpoint. This ensures compatibility with any client, framework (LangChain, LangGraph), or tool that speaks the OpenAI API format.
- **Streaming:** Must support server-sent events (SSE) for low-latency token streaming. Streaming architecture must be designed upfront — not retrofitted.
- **Concurrency:** The NPU is a single-tenant resource. Concurrent requests must be queued or rejected with `429 Too Many Requests` to prevent device lock crashes.
- **Context Management:** Maximum context is 2048 tokens (baked into the HEF). The server must track context usage and handle graceful truncation or rejection when the limit is approached.

## 5. LLM Strategy
- **Scope:** Pure inference — language understanding, reasoning, and generation only.
- **Prompt Templating:** Incoming OpenAI-format messages are rendered into the model's required format (ChatML for Qwen) using the template returned by `llm.prompt_template()`. This is handled server-side; clients send standard message arrays.
- **No Tool Calling:** The server does not process, parse, or relay tool calls. Tool orchestration is fully delegated to LangGraph Agent Services in the layer above.
- **Intended Usage Pattern:** The LLM is used as a *reasoning input* to LangGraph, not as an executor. Typical uses include:
    - **Graph selection:** classifying user intent to determine which LangGraph workflow to invoke.
    - **Decision nodes:** within a graph, calling the LLM to reason about the current state and advise the next step — the graph interprets the response and owns the branching logic.
  In both cases the LLM expresses reasoning in plain language; the graph handles all structured execution. This division plays to the model's strengths and keeps behaviour predictable despite the 1.5B model's limitations.

## 6. Engineering Standards
- **Isolation:** No hardcoded project logic. The server is configuration-driven and project-agnostic.
- **Resilience:** The server must handle NPU error states gracefully — inference timeouts, device faults, and context overflow must produce clean HTTP error responses, not crashes. The service must recover without requiring a manual restart where possible.
- **Observability:** Structured logging for every request: Time-To-First-Token, tokens/sec, context usage, and errors. NPU hardware telemetry (temperature, device status) must be logged periodically and exposed on a health endpoint.
- **Configuration:** Model path, inference parameters (temperature, max tokens), and server settings (port, log level) must be externally configurable via environment variables or a config file.
- **Deployment:** Managed via `systemd` as a persistent background service — starts on boot, restarts on failure.

## 6a. Model Usage Monitoring
- **Approach:** Lightweight SQLite-backed metrics store. Every inference request writes a structured record; no external services required.
- **Data Model:** Each record captures: timestamp, request ID, prompt token count, completion token count, TTFT (ms), tokens/sec, context usage at time of request, model name, and error (if any).
- **Reporting:** A `GET /stats` endpoint on the server returns summary statistics (request count, average/p95 TTFT, average tokens/sec, error rate) for configurable time windows (last hour, last 24h, all time).
- **Future:** LangGraph Agent Services on other devices (Tactical Mesh Pi, etc.) should be evaluated for Langfuse integration, which provides richer session-level and workflow-level tracing suited to the orchestration layer.

## 7. Future Consideration: Home Assistant Integration
A LangGraph Agent Service co-located on this Pi will eventually be needed to bridge the LLM server to Home Assistant. Home Assistant's OS cannot host arbitrary Python services, so the orchestration layer must live here and expose an interface HA can call via REST or webhooks. This is out of scope for the current build and will be addressed as a separate project once the LLM server is stable.
