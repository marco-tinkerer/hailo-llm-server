from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    model_config = {"env_prefix": "NPU_", "env_file": ".env", "env_file_encoding": "utf-8"}

    model_path: Path = Field(
        default=Path("/home/marcomark/hailo-models/Qwen2.5-1.5B-Instruct.hef"),
        description="Path to .hef model file",
    )
    lora_name: str = Field(
        default="",
        description="LoRA adapter name; empty string disables LoRA (required for function-calling HEF)",
    )

    max_generated_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    do_sample: bool = Field(default=False)
    inference_timeout_s: float = Field(default=60.0, ge=1.0)

    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = Field(default="info")
    metrics_db_path: Path = Field(default=Path("metrics.db"), description="SQLite DB for request metrics")


settings = Settings()
