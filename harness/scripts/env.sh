#!/usr/bin/env bash
# Shared environment for the local LLM stack.
LLM_HOME="${LLM_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLAMA_BIN="$LLM_HOME/bin/llama-b11850"
CUDART_DIR="$LLM_HOME/bin/cudart-llama-b11850-bin-ubuntu-cuda-13.4-x64"
# b11985 is used for the 35B MoE: it has --moe-cache-mib (GPU LRU cache for host experts)
LLAMA_BIN_35B="$LLM_HOME/bin/llama-b11985"
CUDART_DIR_35B="$LLM_HOME/bin/cudart-llama-b11985-bin-ubuntu-cuda-13.4-x64"
export LLM_HOME LLAMA_BIN CUDART_DIR LLAMA_BIN_35B CUDART_DIR_35B
export LD_LIBRARY_PATH="$LLAMA_BIN:$CUDART_DIR:$LLAMA_BIN_35B:$CUDART_DIR_35B:${LD_LIBRARY_PATH:-}"
export PATH="$LLAMA_BIN:$PATH"
MODELS="$LLM_HOME/models"
export MODELS
