from typing import Literal, Any
from pydantic import BaseModel, Field
import time
import uuid


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "Llama-3.2-3B-Instruct.hef"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    preserve_context: bool = Field(
        default=False,
        description="If False (default), clears NPU KV-cache before each request for stateless LangGraph/OpenAI calls.",
    )
    tools: list[dict[str, Any]] | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "hailo-10h"
    active: bool = False


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class LoadModelRequest(BaseModel):
    model: str = Field(description="Filename or path of the .hef model to load")
    lora_name: str = Field(default="", description="Optional LoRA adapter name")


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "Llama-3.2-3B-Instruct.hef"
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "Llama-3.2-3B-Instruct.hef"
    choices: list[StreamChoice]


class HealthResponse(BaseModel):
    status: str
    context_used: int
    context_max: int
    context_free: int
    model: str
    lora: str
    version: str = "0.2.0"


class StatsWindow(BaseModel):
    window: str
    request_count: int
    error_count: int
    error_rate: float
    avg_ttft_ms: float | None
    p95_ttft_ms: float | None
    avg_tokens_per_sec: float | None
    avg_completion_tokens: float | None
    total_completion_tokens: int
