#!/usr/bin/env bash
set -euo pipefail

IMAGE=bolt-vllm-nemotron
NAME=vllm

PORT=8080
DETACH=0
KV_CACHE_DTYPE=fp8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    -d|--detach) DETACH=1; shift ;;
    --restart|--restart=*)
      if [[ "$1" == *=* ]]; then RESTART="${1#*=}"; else RESTART="${2:-unless-stopped}"; shift; fi
      shift
      ;;
    --kv-cache-dtype|--kv-cache-dtype=*)
      if [[ "$1" == *=* ]]; then KV_CACHE_DTYPE="${1#*=}"; else KV_CACHE_DTYPE="$2"; shift; fi
      shift
      ;;
    *)
      echo "Usage: $0 [--port N] [-d] [--restart [always|unless-stopped|on-failure]] [--kv-cache-dtype fp16|bf16|fp8|nvfp4]" >&2
      exit 1
      ;;
  esac
done

docker build -t "$IMAGE" .

docker rm -f "$NAME" >/dev/null 2>&1 || true

RUN_ARGS=(docker run --name "$NAME"
  --gpus '"device=1"'
  --ipc=host
  -p "$PORT:8000"
  -v ~/.cache:/root/.cache
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE")

if [[ -n "${RESTART:-}" ]]; then
  RUN_ARGS+=(--restart "$RESTART")
else
  RUN_ARGS+=(--rm)   # --restart and --rm are mutually exclusive
fi

if [[ "$DETACH" == 1 ]]; then
  RUN_ARGS+=(-d)
else
  RUN_ARGS+=(-i --init)
fi

exec "${RUN_ARGS[@]}" "$IMAGE"
