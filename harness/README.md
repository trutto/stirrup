# Reference source

This folder is a **sanitized snapshot of the reference harness** used for the write-up. It is not a
turnkey app: model files, llama.cpp binaries, API keys, `run/`, `logs/` and downloaded models are
not included.

```text
scripts/
  harness.py          HTTP server + phone UI + model manager + agent loop
  ui.html             phone UI
  serve-local.sh      llama.cpp chip runners
  env.sh              binary/model paths
  selftest.sh         API + UI regression test
tests/
  harness_selftest.py, harness_selftest_ui.mjs, harness_agentic.py
  mcp_echo_server.py, test_api.py, test_guards.py, test_units.py
examples/
  stirrup.service     systemd user unit template
  mcp.json            MCP server config example
  agent-policy.json   approvals / auto-think / time budget example
```

## Use it

1. Copy `scripts/*` into a project `llm/scripts/` folder, `tests/*` into `llm/tests/`.
2. Add llama.cpp binaries under `bin/` and GGUFs under `models/`.
3. Create `run/local-llm.key` with your API key, and a systemd user unit from
   `examples/stirrup.service`.
4. Point the UI at the laptop's Tailscale IP (`LLM_HARNESS_BIND`) and open it on the phone.

See the top-level README for architecture, model choices and the full setup guide.

## Placeholders

- `PHONE_SERIAL` is required for `phone_send` — set it to `<phone-tailscale-ip>:5555`.
- `LLM_HARNESS_BIND` defaults to `127.0.0.1`; use your Tailscale IP for the phone.
- No real tokens, keys, IPs or personal paths are in this snapshot.
