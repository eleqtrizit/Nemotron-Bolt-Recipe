#!/usr/bin/env bash
set -euo pipefail

# Needs a GPU, so run at container start, not build time
vllm download-kernels

# Replace the baked-in --kv-cache-dtype with the one requested via KV_CACHE_DTYPE
ARGS=("$@")
NEW=()
SKIP=0
for i in "${!ARGS[@]}"; do
  if [[ $SKIP -eq 1 ]]; then SKIP=0; continue; fi
  case "${ARGS[$i]}" in
    --kv-cache-dtype) SKIP=1; continue ;;          # drop flag and its value
    --kv-cache-dtype=*) continue ;;                # drop inline form
    *) NEW+=("${ARGS[$i]}") ;;
  esac
done
exec "${NEW[@]}" --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
