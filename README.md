# hailo-llm-server

OpenAI-compatible LLM server for the Hailo-10H NPU on Raspberry Pi 5.

Runs Qwen2-1.5B function-calling models on-device via the `hailo_platform.genai` SDK and exposes a `/v1/chat/completions` endpoint compatible with any OpenAI client.

## Hardware

- Raspberry Pi 5
- Hailo-10H M.2 AI accelerator
- `h10-hailort` 5.1.1 and `python3-h10-hailort` installed system-wide

## Performance

| Metric | Value |
|---|---|
| Time to first token (TTFT) | ~0.4–0.8s |
| Decode speed | ~7–10 tok/s |
| Max context | 2048 tokens |

## Setup

```bash
git clone https://github.com/marco-tinkerer/hailo-llm-server
cd hailo-llm-server/src/npu_server
uv venv --system-site-packages
uv sync
```

## Configuration

All settings use `NPU_` env vars or a `.env` file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `NPU_MODEL_PATH` | `/home/marcomark/hailo-models/Qwen2-1.5B-Instruct-Function-Calling-v1.hef` | Path to `.hef` model |
| `NPU_LORA_NAME` | `huggingface_lora_adapter` | LoRA adapter name (empty string disables) |
| `NPU_MAX_GENERATED_TOKENS` | `512` | Default max tokens per response |
| `NPU_TEMPERATURE` | `0.0` | Sampling temperature |
| `NPU_INFERENCE_TIMEOUT_S` | `60.0` | Seconds before stalled inference returns 503 |
| `NPU_SERVER_HOST` | `0.0.0.0` | Bind address |
| `NPU_SERVER_PORT` | `8080` | Bind port |
| `NPU_LOG_LEVEL` | `info` | Log level |
| `NPU_METRICS_DB_PATH` | `metrics.db` | SQLite metrics database path |

## Running

```bash
cd src/npu_server
.venv/bin/python main.py
```

## Install as a service

```bash
sudo cp hailo-llm-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-llm-server
sudo journalctl -u hailo-llm-server -f
```

## API

### Health

```bash
curl http://localhost:8080/health
```

### Chat completion (non-streaming)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"npu-qwen2","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

### Chat completion (streaming)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"npu-qwen2","messages":[{"role":"user","content":"Count from 1 to 5."}],"stream":true}'
```

### With tools

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "npu-qwen2",
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

### Usage stats

```bash
curl "http://localhost:8080/stats?window=24h"
```

```json
{
  "window": "24h",
  "request_count": 42,
  "error_count": 0,
  "error_rate": 0.0,
  "avg_ttft_ms": 412.5,
  "p95_ttft_ms": 680.2,
  "avg_tokens_per_sec": 8.3,
  "avg_completion_tokens": 47.1,
  "total_completion_tokens": 1978
}
```

### Clear context

```bash
curl -X POST http://localhost:8080/v1/context/clear
```

## Notes

- Single-tenant: only one concurrent request is served. Additional requests receive `429`.
- Context window is 2048 tokens (baked into the HEF). Overflow requests return `400`.
- On `HailoRTException` the server attempts one automatic re-initialization before returning `503`.
- NPU chip temperature is logged every 60 seconds.
