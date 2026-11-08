#!/usr/bin/env bash
# Local llama.cpp servers for DSH. Binds the tailnet IP, requires an API key.
# Usage: serve-local.sh {start|stop|status|run} [9b|27b|9b-dflash|both|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
DSH_HOME="${DSH_HOME:-$HOME/.dsh}"
RUNDIR="$LLM_HOME/run"; LOGDIR="$LLM_HOME/logs"
mkdir -p "$RUNDIR" "$LOGDIR"

KEY="${LOCAL_LLM_KEY:-}"
if [[ -z "$KEY" && -f "$LLM_HOME/run/local-llm.key" ]]; then
  KEY="$(tr -d '\n' < "$LLM_HOME/run/local-llm.key")"
fi
if [[ -z "$KEY" && -f "$DSH_HOME/.env" ]]; then
  KEY="$(grep -E '^LOCAL_LLM_KEY=' "$DSH_HOME/.env" | tail -1 | cut -d= -f2- || true)"
fi
[[ -n "$KEY" ]] || { echo "LOCAL_LLM_KEY missing: run llm/dsh/install.sh first" >&2; exit 1; }

IP="${LLM_IP:-}"
[[ -n "$IP" ]] || IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
[[ -n "$IP" ]] || IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || true)"
[[ -n "$IP" ]] || { echo "cannot detect host IP; set LLM_IP" >&2; exit 1; }
BIND_IP="${BIND_IP:-$IP}"   # proxy runs backends on 127.0.0.1

# Daily driver (set 2026-11-01): Qwen3.8 35B-A3B distill IQ2_M - the only local model
# left on disk (2026-11-01 cleanup). Benchmarked on this box at 43.4 tok/s gen @32k /
# 36.7 @128k, 11/11 scored, vs the dense 27B IQ2_XXS (8.2 tok/s, 9/11) and IQ3_M
# (9.7 tok/s, memory cliff). Downloadable again from unsloth/Qwen3.8-27B-GGUF-style repos
# if a different backend is ever wanted back.
MODEL_35B="${MODEL_35B_PATH:-$MODELS/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf}"

run_35b() {
  # hybrid attention: only every 4th layer has a KV cache, so long ctx is cheap with q4_0 KV.
  # NOTE: the 2026-11-01 distill backend is NOT fully hybrid - 256k ctx collapsed to
  # 13.6 tok/s (fits fewer GPU layers); see HARNESS.md for the measured ctx/speed curve.
  exec "${LLAMA_BIN_35B:-$LLAMA_BIN}/llama-server" -m "$MODEL_35B" --alias local-35b --host "$BIND_IP" --port "${PORT_35B:-8084}" --api-key "$KEY" \
    -c "${CTX35B:-32768}" -fa on -ctk "${KV35B:-q4_0}" -ctv "${KV35B:-q4_0}" -ub "${UB35B:-1024}" -t 8 -np 1 --reasoning "${LLM_REASONING:-auto}" --reasoning-format "${LLM_REASONING_FORMAT:-deepseek}" --jinja --fit on -lm mmap \
    --moe-cache-mib "${MOECACHE35B:-2048}"
}

MODEL_GROOT="${MODEL_GROOT_PATH:-$MODELS/Groot-19B-A3B-TTQ-3.56bpw-IQ3_XXS.gguf}"
GROOT_BIN="${LLAMA_BIN_GROOT:-$LLM_HOME/bin/llama-b12189}"
GROOT_CUDART="${CUDART_DIR_GROOT:-$LLM_HOME/bin/cudart-llama-b12189-bin-ubuntu-cuda-13.5-x64}"

run_groot() {
  # Groot-19B-A3B (TTQ IQ3_XXS): ~63 tok/s decode / ~1800 tok/s prefill on the 5050,
  # 10/11 bench, 6/7 agent suite. Added 2026-11-07 as the phone app default.
  [[ -f "$MODEL_GROOT" ]] || { echo "missing $MODEL_GROOT" >&2; exit 1; }
  export LD_LIBRARY_PATH="$GROOT_BIN:$GROOT_CUDART:${LD_LIBRARY_PATH:-}"
  exec "$GROOT_BIN/llama-server" -m "$MODEL_GROOT" --alias local-groot --host "$BIND_IP" --port "${PORT_GROOT:-9083}" --api-key "$KEY" \
    -c "${CTX_GROOT:-65536}" -fa on -ctk "${KV_GROOT:-q4_0}" -ctv "${KV_GROOT:-q4_0}" -ub "${UB_GROOT:-1024}" -t 8 -np 1 --jinja --fit on -lm mmap \
    --moe-cache-mib "${MOECACHE_GROOT:-2048}"
}

run_dory() {
  local BIN="$LLM_HOME/bin/llama-dory-98a74c9"
  local MODEL_DORY="${MODEL_DORY_PATH:-$MODELS/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf}"
  [[ -f "$MODEL_DORY" ]] || { echo "missing $MODEL_DORY" >&2; exit 1; }
  [[ -x "$BIN/llama-server" ]] || { echo "missing $BIN/llama-server" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_DORY" --alias local-dory-4b --host "$BIND_IP" --port "${PORT_DORY:-9085}" --api-key "$KEY" -c "${CTX_DORY:-65536}" -ngl 999 -np 1 -ub 512 -t 8 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --override-kv "dory.num_loops=int:${DORY_LOOPS:-2},dory.recurrent_kv_cache_mode=str:${DORY_KV:-last_loop}" "$@"
}

run_lfm3() {
  local MODEL_LFM3="${MODEL_LFM3_PATH:-$MODELS/LFM3-1.7B-Q4_K_M.gguf}"
  local BIN="${LLAMA_BIN_35B:-$LLAMA_BIN}"
  [[ -f "$MODEL_LFM3" ]] || { echo "missing $MODEL_LFM3" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:${CUDART_DIR_35B:-$CUDART_DIR}:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_LFM3" --alias local-lfm3 --host "$BIND_IP" --port "${PORT_LFM3:-9086}" --api-key "$KEY" -c "${CTX_LFM3:-65536}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np "${NP_LFM3:-1}" --jinja --fit on -lm mmap "$@"
}

run_emp4b() {
  local MODEL_EMP4B="${MODEL_EMP4B_PATH:-$MODELS/Qwen3.8-4B-Distill-Q4_K_M.gguf}"
  local MMPROJ_EMP4B="${MMPROJ_EMP4B:-$MODELS/Qwen3.5-4B-mmproj-F16.gguf}"
  local BIN="${LLAMA_BIN_35B:-$LLAMA_BIN}"
  [[ -f "$MODEL_EMP4B" ]] || { echo "missing $MODEL_EMP4B" >&2; exit 1; }
  local MM=()
  if [ -f "$MMPROJ_EMP4B" ]; then
    MM=(--mmproj "$MMPROJ_EMP4B")
  fi
  export LD_LIBRARY_PATH="$BIN:${CUDART_DIR_35B:-$CUDART_DIR}:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_EMP4B" "${MM[@]}" --alias local-emp4b --host "$BIND_IP" --port "${PORT_EMP4B:-9087}" --api-key "$KEY" -c "${CTX_EMP4B:-32768}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --jinja --fit on -lm mmap "$@"
}

run_emp9b() {
  local MODEL_EMP9B="${MODEL_EMP9B_PATH:-$MODELS/Qwen3.8-9B-Distill-Q4_K_M.gguf}"
  local BIN="${LLAMA_BIN_35B:-$LLAMA_BIN}"
  [[ -f "$MODEL_EMP9B" ]] || { echo "missing $MODEL_EMP9B" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:${CUDART_DIR_35B:-$CUDART_DIR}:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_EMP9B" --alias local-emp9b --host "$BIND_IP" --port "${PORT_EMP9B:-9088}" --api-key "$KEY" -c "${CTX_EMP9B:-65536}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --jinja --fit on -lm mmap "$@"
}

run_metis() {
  local MODEL_METIS="${MODEL_METIS_PATH:-$MODELS/Metis-1.6-Praxis-Q4_K_M.gguf}"
  local BIN="${LLAMA_BIN_METIS:-$LLM_HOME/bin/llama-b12110}"
  local CUD="${CUDART_METIS:-$LLM_HOME/bin/cudart-llama-b12110-bin-ubuntu-cuda-13.4-x64}"
  [[ -f "$MODEL_METIS" ]] || { echo "missing $MODEL_METIS" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:$CUD:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_METIS" --alias local-metis --host "$BIND_IP" --port "${PORT_METIS:-9089}" --api-key "$KEY" -c "${CTX_METIS:-65536}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --jinja --fit on -lm mmap "$@"
}

run_flint() {
  local MODEL_FLINT="${MODEL_FLINT_PATH:-$MODELS/flint-UD-IQ2_XXS.gguf}"
  local BIN="${LLAMA_BIN_FLINT:-${LLAMA_BIN_35B:-$LLAMA_BIN}}"
  local CUD="${CUDART_FLINT:-${CUDART_DIR_35B:-$CUDART_DIR}}"
  [[ -f "$MODEL_FLINT" ]] || { echo "missing $MODEL_FLINT" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:$CUD:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_FLINT" --alias local-flint --host "$BIND_IP" --port "${PORT_FLINT:-9092}" --api-key "$KEY" -c "${CTX_FLINT:-32768}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --reasoning auto --reasoning-format deepseek --jinja --fit on -lm mmap --moe-cache-mib "${MOECACHE_FLINT:-2048}" "$@"
}

run_apex() {
  local MODEL_APEX="${MODEL_APEX_PATH:-$MODELS/Qwen3.8-35B-A3B-Distill-APEX-Mini.gguf}"
  local BIN="${LLAMA_BIN_APEX:-${LLAMA_BIN_35B:-$LLAMA_BIN}}"
  local CUD="${CUDART_APEX:-${CUDART_DIR_35B:-$CUDART_DIR}}"
  [[ -f "$MODEL_APEX" ]] || { echo "missing $MODEL_APEX" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:$CUD:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_APEX" --alias local-apex --host "$BIND_IP" --port "${PORT_APEX:-9093}" --api-key "$KEY" -c "${CTX_APEX:-32768}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --reasoning auto --reasoning-format deepseek --jinja --fit on -lm mmap --moe-cache-mib "${MOECACHE_APEX:-2048}" "$@"
}

run_dynamic() {
  # Models downloaded from Hugging Face through the app: one shared lane (:8099/9099).
  local BIN="${LLAMA_BIN_DYNAMIC:-$LLM_HOME/bin/llama-b12189}"
  local CUD="${CUDART_DYNAMIC:-$LLM_HOME/bin/cudart-llama-b12189-bin-ubuntu-cuda-13.5-x64}"
  local MODEL="${MODEL_DYNAMIC_PATH:-}"
  [[ -n "$MODEL" ]] || { echo "MODEL_DYNAMIC_PATH not set" >&2; exit 1; }
  [[ -f "$MODEL" ]] || { echo "missing $MODEL" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:$CUD:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL" --alias local-downloaded --host "$BIND_IP" \
    --port "${PORT_DYNAMIC:-9099}" --api-key "$KEY" -c "${CTX_DYNAMIC:-32768}" \
    -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np 1 --jinja --fit on -lm mmap "$@"
}

run_bonsai() {
  local MODEL_BONSAI="${MODEL_BONSAI_PATH:-$MODELS/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
  local BIN="${LLAMA_BIN_BONSAI:-$LLM_HOME/bin/llama-prism}"
  [[ -x "$BIN/llama-server" ]] || { echo "missing $BIN/llama-server" >&2; exit 1; }
  [[ -f "$MODEL_BONSAI" ]] || { echo "missing $MODEL_BONSAI" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_BONSAI" --alias local-bonsai --host "$BIND_IP" --port "${PORT_BONSAI:-9100}" --api-key "$KEY" -ngl 999 -c "${CTX_BONSAI:-16384}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np "${NP_BONSAI:-1}" --reasoning-effort "${REASONING_EFFORT_BONSAI:-medium}" --jinja "$@"
}

run_ds9() {
  local MODEL_DS9="${MODEL_DS9_PATH:-$MODELS/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf}"
  local BIN="${LLAMA_BIN_DS9:-${LLAMA_BIN_35B:-$LLAMA_BIN}}"
  local CUD="${CUDART_DS9:-${CUDART_DIR_35B:-$CUDART_DIR}}"
  [[ -f "$MODEL_DS9" ]] || { echo "missing $MODEL_DS9" >&2; exit 1; }
  export LD_LIBRARY_PATH="$BIN:$CUD:${LD_LIBRARY_PATH:-}"
  exec "$BIN/llama-server" -m "$MODEL_DS9" --alias local-ds9 --host "$BIND_IP" --port "${PORT_DS9:-9094}" --api-key "$KEY" -c "${CTX_DS9:-32768}" -fa on -ctk q4_0 -ctv q4_0 -ub 1024 -t 8 -np "${NP_DS9:-1}" --jinja --fit on -lm mmap "$@"
}

pidfile() { echo "$RUNDIR/local-$1.pid"; }
is_up() {
  local p; p=$(pidfile "$1")
  if [[ -f "$p" ]] && kill -0 "$(cat "$p")" 2>/dev/null; then return 0; fi
  local port; port=$(port_of "$1")
  curl -s -m 2 -o /dev/null -H "Authorization: Bearer $KEY" "http://$IP:$port/health" 2>/dev/null
}
port_of() { case "$1" in 35b) echo 8084;; *) echo 0;; esac; }

start_one() { # mode
  local mode="$1"
  if is_up "$mode"; then echo "local-$mode already running (pid $(cat "$(pidfile "$mode")" 2>/dev/null || echo '?'))"; return 0; fi
  nohup "$0" run "$mode" >>"$LOGDIR/local-$mode.log" 2>&1 &
  echo $! > "$(pidfile "$mode")"
  sleep 1
  if is_up "$mode"; then
    echo "local-$mode started on http://$IP:$(port_of "$mode") (pid $(cat "$(pidfile "$mode")"))"
  else
    echo "local-$mode failed; see $LOGDIR/local-$mode.log" >&2; tail -5 "$LOGDIR/local-$mode.log" >&2; return 1
  fi
}

stop_one() {
  local mode="$1" p; p=$(pidfile "$mode")
  if [[ -f "$p" ]]; then kill "$(cat "$p")" 2>/dev/null || true; rm -f "$p"; echo "local-$mode stopped"; else echo "local-$mode not running"; fi
}

status_one() {
  local mode="$1" port; port=$(port_of "$mode")
  if is_up "$mode"; then
    local n; n=$(curl -s -m 3 -H "Authorization: Bearer $KEY" "http://$IP:$port/v1/models" | jq -r '.data[0].id // "?"' 2>/dev/null || echo "?")
    echo "local-$mode  UP   http://$IP:$port  model=$n"
  else echo "local-$mode  down"; fi
}

mode_start() { start_one "$1"; }
mode_stop()  { stop_one "$1"; }

action="${1:-start}"; which="${2:-35b}"
case "$action" in
  run)   case "$which" in 35b) run_35b;; groot) run_groot;; dory) run_dory;; lfm3) run_lfm3;; emp4b) run_emp4b;; emp9b) run_emp9b;; metis) run_metis;; flint) run_flint;; apex) run_apex;; ds9) run_ds9;; bonsai) run_bonsai;; dynamic) run_dynamic;; *) echo "run needs 35b|groot|dory|lfm3|emp4b|emp9b|metis|flint|apex|ds9|bonsai|dynamic" >&2; exit 2;; esac;;
  start) mode_start "$which";;
  stop)  mode_stop "$which";;
  proxy)
         # the harness service replaces llm-proxy.py (phone UI :8090)
         case "$which" in
           start)   systemctl --user start local-llm-harness.service; echo "harness started (:8084 + phone UI :8090)";;
           stop)    systemctl --user stop local-llm-harness.service; echo "harness stopped";;
           status)
             printf ':8084 '; curl -s -m 2 "http://$IP:8084/health" | jq -c . 2>/dev/null || echo down
             ;;
           *) echo "proxy needs start|stop|status" >&2; exit 2;;
         esac;;
  use)   if systemctl --user is-active --quiet local-llm-harness.service 2>/dev/null; then
           echo "the harness service manages the servers. Use 'proxy stop' first for direct mode." >&2
           exit 1
         fi
         stop_one 35b
         pkill -x llama-server 2>/dev/null || true
         sleep 2
         case "$which" in
           moe|35b) start_one 35b;;
           *) echo "use needs moe (only local model left)" >&2; exit 2;;
         esac;;
  status) status_one 35b;;
  *) echo "usage: serve-local.sh {start|stop|use|status|run|proxy} [groot|moe|35b]" >&2; exit 2;;
esac