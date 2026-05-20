"""Basic inference smoke test — runs outside FastAPI to confirm NPU generation works."""
import asyncio
import logging
import time
from jinja2 import Environment
import json

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

from npu_server.config import settings
from npu_server.provider import HailoProvider


async def main() -> None:
    provider = HailoProvider(settings)
    provider.initialize()

    template = provider.prompt_template()
    env = Environment()
    env.filters["tojson"] = json.dumps

    messages = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    prompt = env.from_string(template).render(messages=messages, add_generation_prompt=True)

    print("\n--- Prompt ---")
    print(prompt)
    print("\n--- Response (streaming) ---")

    start = time.monotonic()
    token_count = 0
    async with provider.generate(prompt) as stream:
        async for token in stream:
            print(token, end="", flush=True)
            token_count += 1
    elapsed = time.monotonic() - start

    print(f"\n\n--- Stats ---")
    print(f"Tokens: {token_count}, Elapsed: {elapsed:.2f}s, Tok/s: {token_count/elapsed:.1f}")
    print(f"Context usage after: {provider.get_context_usage()} / {provider.max_context_capacity()} tokens")

    provider.release()


if __name__ == "__main__":
    asyncio.run(main())
