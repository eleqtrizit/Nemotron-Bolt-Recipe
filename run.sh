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
  --gpu SPEC              GPUs to expose: bare indices ("0", "0,1") or a docker
                          --gpus spec ("device=1", "all"). Default: all.
  --tp N                  Tensor parallelism. Default 1. TP=2 is NOT supported by
                          this checkpoint (IQ2_XXS dense weights); opt-in for tests.
  --kv-cache-dtype DTYPE  fp8 (default), bf16, nvfp4. nvfp4 is blocked upstream for
                          this model (see AGENTS.md).
  -h, --help              Show this help.

Examples:
  ./run.sh                          # foreground, port 8080, all GPUs, TP=1
  ./run.sh --port 9000              # different host port
  ./run.sh --gpu 1                  # pin to second GPU, TP=1
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

# Default: expose all GPUs. docker run gives NO GPUs without --gpus.
GPU="${GPU:-all}"

# Pre-flight: refuse to start when the serving GPU(s) can't satisfy the
# gpu-memory-utilization baked into the Dockerfile (default 0.96), instead of
# dying 2 minutes into engine init with a cryptic ValueError.
PREFLIGHT_UTIL=$(grep -oE 'gpu-memory-utilization", "[0-9.]+' Dockerfile | rg -o '[0-9.]+$')
PREFLIGHT_UTIL="${PREFLIGHT_UTIL:-0.96}"
preflight_gpu_memory() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  [[ "$GPU" == "all" ]] && GPU_LIST=$(nvidia-smi -L 2>/dev/null | rg -o -P 'GPU \K[0-9]+' | tr '
' ',' | sed 's/,$//') || GPU_LIST="${GPU#device=}"
  [[ -z "$GPU_LIST" ]] && return 0
  # TP=1 only uses the first visible GPU as cuda:0; TP>1 uses them all
  [[ "$TP" -ge 2 ]] || GPU_LIST="${GPU_LIST%%,*}"
  local idx total free need pct name
  for idx in ${GPU_LIST//,/ }; do
    total=$(nvidia-smi -i "$idx" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    free=$(nvidia-smi -i "$idx" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [[ -z "$total" || -z "$free" ]] && continue
    need=$(python3 -c "print(int($total * $PREFLIGHT_UTIL))" 2>/dev/null) || continue
    if (( free < need )); then
      pct=$(python3 -c "print(round(100.0 * $free / $total, 1))")
      echo "" >&2
      echo "ERROR: GPU $idx doesn't have enough free memory to serve this model." >&2
      echo "  Needs ~$((need / 1024)) GiB free (--gpu-memory-utilization $PREFLIGHT_UTIL of $((total / 1024)) GiB)," >&2
      echo "  but only $((free / 1024)) GiB ($pct%) is free. Something else is using the card." >&2
      echo "" >&2
      echo "Fix one of:" >&2
      echo "  1. Free the GPU (stop the process using it), or" >&2
      echo "  2. Pin an unused card:  ./run.sh --gpu <index>" >&2
      echo "     GPUs by free memory:" >&2
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null \
        | sort -t, -k2 -nr | sed 's/^/       /' | head -4 >&2
      exit 1
    fi
  done
}
preflight_gpu_memory

docker build -t "$IMAGE" .

docker rm -f "$NAME" >/dev/null 2>&1 || true

RUN_ARGS=(docker run --name "$NAME"
  --ipc=host
  -p "$PORT:8000"
  -v ~/.cache:/root/.cache
  -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE"
  -e TP="$TP")

RUN_ARGS+=(--gpus "$GPU")

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
