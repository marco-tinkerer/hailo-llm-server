# NPU LLM Server

OpenAI-compatible HTTP server for the Hailo-10H NPU running Qwen2.5 function-calling models.

## Requirements

- Raspberry Pi 5 with Hailo-10H M.2 module
- `h10-hailort` 5.1.1 and `python3-h10-hailort` installed system-wide
- `uv` package manager

## Setup

```bash
cd src/npu_server
uv venv --system-site-packages
uv sync
```

## Configuration

All settings are read from environment variables prefixed `NPU_` or from a `.env` file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `NPU_MODEL_PATH` | `/home/marcomark/hailo-models/Qwen2-1.5B-Instruct-Function-Calling-v1.hef` | Path to `.hef` model |
| `NPU_LORA_NAME` | `huggingface_lora_adapter` | LoRA adapter name (empty string disables LoRA) |
| `NPU_MAX_GENERATED_TOKENS` | `512` | Default max tokens per response |
| `NPU_TEMPERATURE` | `0.0` | Sampling temperature |
| `NPU_DO_SAMPLE` | `false` | Enable sampling (vs greedy decoding) |
| `NPU_INFERENCE_TIMEOUT_S` | `60.0` | Seconds before a stalled inference returns 503 |
| `NPU_SERVER_HOST` | `0.0.0.0` | Bind address |
| `NPU_SERVER_PORT` | `8080` | Bind port |
| `NPU_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warning`, `error`) |

Example `.env`:
```
NPU_MAX_GENERATED_TOKENS=256
NPU_LOG_LEVEL=debug
```

## Running

```bash
cd src/npu_server
.venv/bin/python main.py
```

## Install as systemd service

```bash
sudo cp hailo-llm-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hailo-llm-server
sudo journalctl -u hailo-llm-server -f
```

## API

### Health check

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "ok",
  "context_used": 0,
  "context_max": 2048,
  "context_free": 2048,
  "model": "Qwen2-1.5B-Instruct-Function-Calling-v1.hef",
  "lora": "huggingface_lora_adapter"
}
```

### Non-streaming completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "npu-qwen2",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

### Streaming completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "npu-qwen2",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
```

### With tools

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "npu-qwen2",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
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

### Clear conversation context

The NPU maintains context between requests. Call this to reset it:

```bash
curl -X POST http://localhost:8080/v1/context/clear
```

## Notes

- Only one concurrent request is served (NPU is single-tenant). Concurrent requests receive `429`.
- Context window is 2048 tokens (baked into the HEF). Requests that would overflow return `400`.
- The model uses ChatML format rendered via the HEF's embedded Jinja2 template.
- NPU temperature is logged every 60 seconds at `INFO` level.
- On `HailoRTException` the server attempts one automatic re-initialization before returning `503`.
