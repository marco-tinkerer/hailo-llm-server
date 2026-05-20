# Feature Proposal: MCP Server for NPU Monitoring

## Summary

A minimal MCP server that exposes the NPU server's observability endpoints as tools, so an agent or Claude Desktop can check hardware health and usage stats in conversation.

## Proposed Tools

| Tool | Backing endpoint | Purpose |
|---|---|---|
| `npu_health` | `GET /health` | Device status, context usage, model loaded |
| `npu_stats` | `GET /stats?window=` | Request count, TTFT, tok/s, error rate |

## Proposed Stack

- `mcp` Python SDK (stdio transport)
- `httpx` HTTP client
- `NPU_SERVER_URL` env var (default `http://localhost:8080`)

## Open Questions

- Should this live in this repo (`src/npu_mcp`) or a separate repo?
- Is the target client Claude Desktop specifically, or does it need to be usable by agents too?
- Should temperature telemetry be surfaced as a third tool, or is it sufficiently covered by the log?
- Is stdio transport sufficient, or is SSE needed for remote access?
