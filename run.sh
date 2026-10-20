#!/usr/bin/env bash
set -euo pipefail

IMAGE=bolt-vllm-nemotron
NAME=vllm

PORT=8080
DETACH=0
KV_CACHE_DTYPE=fp8
GPU=""
TP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    -d|--detach) DETACH=1; shift ;;
    --gpu|--gpu=*)
      if [[ "$1" == *=* ]]; then GPU="${1#*=}"; else GPU="$2"; shift; fi
      shift
      ;;
    --tp|--tp=*)
      if [[ "$1" == *=* ]]; then TP="${1#*=}"; else TP="$2"; shift; fi
      shift
      ;;
    --restart|--restart=*)
      if [[ "$1" == *=* ]]; then RESTART="${1#*=}"; else RESTART="${2:-unless-stopped}"; shift; fi
      shift
      ;;
    --kv-cache-dtype|--kv-cache-dtype=*)
      if [[ "$1" == *=* ]]; then KV_CACHE_DTYPE="${1#*=}"; else KV_CACHE_DTYPE="$2"; shift; fi
      shift
      ;;
    -h|--help)
      cat <<USAGE
Usage: ./run.sh [OPTIONS]

Build the bolt-vllm-nemotron image and serve
nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB with b12x + FlashInfer + dflash.

Options:
  --port N                Host port to expose (default: 8080); maps to vLLM port 8000.
  -d, --detach            Run detached; logs via 'docker logs -f vllm'.
  --restart [POLICY]      Docker restart policy: always, unless-stopped, on-failure.
                          Default unless-stopped. Disables --rm (Docker forbids both).
  --gpu SPEC              Pin GPUs, e.g. "device=1" or "all". Default: all GPUs
                          (docker run gives NO GPUs unless --gpus is passed).
  --tp N                  Tensor parallelism. Default: auto -- 2 when the host has
                          2+ GPUs and no --gpu pin is given, otherwise 1.
  --kv-cache-dtype DTYPE  fp8 (default), bf16, nvfp4. nvfp4 is blocked upstream for
                          this model (see AGENTS.md).
  -h, --help              Show this help.

Examples:
  ./run.sh                          # foreground, port 8080, TP auto-detected
  ./run.sh --port 9000              # different host port
  ./run.sh --gpu "device=1"         # pin to second GPU, TP=1
  ./run.sh -d --restart always      # detached, restart on crash and daemon boot
  ./run.sh --kv-cache-dtype bf16    # full-precision KV (needs lower max-model-len)
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run './run.sh --help' for usage." >&2
      exit 1
      ;;
  esac
done

# Auto-detect GPU count: two+ GPUs with no explicit pin -> tensor parallelism 2.
# NOTE: docker run without --gpus exposes NO GPUs (unless nvidia is the default
# runtime), so "all GPUs" must be passed explicitly whenever no pin is given.
if [[ -z "$TP" ]]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | grep -c 'GPU ' || true)
  if [[ -n "$GPU" ]]; then
    TP=1
  elif [[ "$NGPU" -ge 2 ]]; then
    TP=2
  else
    TP=1
  fi
fi

docker build -t "$IMAGE" .

docker rm -f "$NAME" >/dev/null 2>&1 || true

RUN_ARGS=(docker run --name "$NAME"
  --ipc=host
  -p "$PORT:8000"
  -v ~/.cache:/root/.cache
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE"
  -e TP="$TP")

if [[ -n "$GPU" ]]; then
  RUN_ARGS+=(--gpus "\"$GPU\"")
else
  RUN_ARGS+=(--gpus all)
fi

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
