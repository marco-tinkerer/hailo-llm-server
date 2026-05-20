# Phase 0: Discovery Findings

## 1. Available Models

| File | Size | Status | Notes |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct.hef` | 2.3GB | Working | Base model, ignores tool calls |
| `Qwen2-1.5B-Instruct-Function-Calling-v1.hef` | 3.0GB | Working (with flag) | LoRA adapter model, requires `lora_name='huggingface_lora_adapter'` |

**Active model choice:** `Qwen2-1.5B-Instruct-Function-Calling-v1.hef` for tool-calling use cases.

**Important:** Without `lora_name`, the function-calling HEF throws `HAILO_INTERNAL_FAILURE(8)`.

---

## 2. SDK Interface

### Loading
```python
from hailo_platform.genai import LLM, VDevice

vdevice = VDevice()
llm = LLM(
    vdevice,
    '/home/marcomark/hailo-models/Qwen2-1.5B-Instruct-Function-Calling-v1.hef',
    lora_name='huggingface_lora_adapter'
)
```

### Streaming (primary mode)
```python
with llm.generate(prompt_str, max_generated_tokens=200, do_sample=False) as gen:
    for token in gen:
        # token is a str fragment, NOT bytes
        print(token, end='', flush=True)
```

- `generate()` is a **context manager** that yields individual `str` token fragments
- `generate_all()` returns the full response as a single `str`
- **Stop tokens (`<|im_end|>`, `<|endoftext|>`) ARE yielded as tokens** — they must be stripped
- The SDK maintains conversation context between calls; use `llm.clear_context()` to reset
- **`generate()` does NOT accept a `tools` kwarg** — tools must be baked into the prompt string

### Context
- `max_context_capacity()` → **2048 tokens** (baked into the HEF, not configurable)
- `get_context_usage_size()` → current token count in context
- `prompt_template()` → returns the Jinja2 chat template baked into the model

---

## 3. Prompt Templating

The SDK **does not handle tools automatically**. The caller must:
1. Call `llm.prompt_template()` to get the Jinja2 template
2. Render it manually with `messages` and `tools` using `jinja2.Environment`
3. Pass the rendered raw string to `llm.generate()`

```python
from jinja2 import Environment
import json

env = Environment()
env.filters['tojson'] = json.dumps  # required — the template uses this filter
template = env.from_string(llm.prompt_template())

rendered = template.render(
    messages=[{"role": "user", "content": "..."}],
    tools=[...],           # omit or None for no-tool prompts
    add_generation_prompt=True
)
```

### Tool call format (in rendered prompt)
```
<tool_call>
{"name": "function_name", "arguments": {"key": "value"}}
</tool_call>
```

### Tool response format (in rendered prompt)
Tool responses are wrapped in a `user` turn:
```
<|im_start|>user
<tool_response>
{result content}
</tool_response><|im_end|>
```

---

## 4. Tool Call Output: Observed Behavior

This is the most important discovery. The model's actual output is **inconsistent** with the prompt template.

### Expected (per template)
```
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
```

### Actual observed outputs (varies per call)

**Format A — markdown code fence (most common):**
```
```json
{"name": "get_weather", "arguments": {"location": "Paris"}}
```
```

**Format B — raw JSON (no wrapper):**
```
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
```

**Format C — stop token inline:** all formats end with `<|im_end|>` as the last token in the stream.

### Reliability issues observed
- Model called a wrong tool when no tool was needed ("What is the capital of France?" → called `calculate` with `expression: "capital of France"`)
- After receiving a tool result, model called the same tool again instead of synthesizing an answer
- These are characteristic 1.5B model reliability issues the repairing parser must handle

---

## 5. Performance Baseline

| Metric | Value |
|---|---|
| Time to First Token (TTFT) | ~0.4–0.8s |
| Tokens/sec (decode) | ~7–10 tok/s |
| Max context | 2048 tokens |
| HailoRT version | 5.1.1 |
| Firmware | 5.1.1 (HAILO10H) |

---

## 6. Implications for Implementation Plan

| Plan Assumption | Reality | Impact |
|---|---|---|
| SDK handles prompt templating natively | Must render Jinja2 manually | Phase 2: add template rendering layer |
| `<tool_call>` XML tag detection | Model outputs markdown fences or raw JSON | Phase 3: parser must handle 3+ output formats |
| Tool calling works reliably | Model confuses tool calls with normal responses | Phase 3: add tool-call vs plain-text classifier |
| Jinja2 as a dependency for templates | Still needed for manual rendering | Phase 1: include jinja2 in dependencies |
| `generate()` yields clean tokens | Stop tokens appear in stream | Strip `<|im_end|>` and `<|endoftext|>` from output |
| Single model (Qwen2.5-1.5B-Instruct) | Function-calling requires separate HEF + lora_name | Config needs two model paths |
