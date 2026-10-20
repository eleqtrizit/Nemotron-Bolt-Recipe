#!/usr/bin/env bash
set -euo pipefail

# Needs a GPU, so run at container start, not build time
vllm download-kernels

# Rewrite tensor-parallel-size and kv-cache-dtype from env (set by run.sh);
# defaults keep the flags from the Dockerfile CMD.
ARGS=("$@")
NEW=()
rewrite_kv=0
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
  case "${ARGS[$i]}" in
    --tensor-parallel-size)
      NEW+=(--tensor-parallel-size "${TP:-1}")
      ((i+=2)) || true
      ;;
    --tensor-parallel-size=*)
      NEW+=(--tensor-parallel-size "${TP:-1}")
      ((i+=1)) || true
      ;;
    --kv-cache-dtype)
      NEW+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}")
      ((i+=2)) || true
      ;;
    --kv-cache-dtype=*)
      NEW+=(--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}")
      ((i+=1)) || true
      ;;
    *) NEW+=("${ARGS[$i]}"); ((i+=1)) || true ;;
  esac
done

exec "${NEW[@]}"
