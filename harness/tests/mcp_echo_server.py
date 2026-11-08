#!/usr/bin/env python3
"""Minimal MCP stdio server used to test the harness MCP client (one `echo` tool)."""
import json
import sys


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except ValueError:
        continue
    method = req.get("method")
    mid = req.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "harness-echo", "version": "0.1"}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
            "name": "echo",
            "description": "Echo the given text back, prefixed with MCP ECHO.",
            "inputSchema": {"type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"]}}]}})
    elif method == "tools/call":
        args = (req.get("params") or {}).get("arguments") or {}
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [
            {"type": "text", "text": "MCP ECHO: " + str(args.get("text", ""))}]}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "unknown method"}})
