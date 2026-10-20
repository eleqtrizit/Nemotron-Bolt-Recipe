#!/usr/bin/env bash
# Integration tests for run.sh GPU selection + startup + memory usage.
# Runs unattended; prints PASS/FAIL per step and exits non-zero on any failure.
set -uo pipefail

PORT=8050
STARTUP_TIMEOUT=1200   # seconds to wait for "Application startup complete" (cold starts are slow)
POLL_INTERVAL=10
LOGDIR=$(mktemp -d /tmp/bolt-tests.XXXXXX)
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
MIN_LOAD_FRACTION=0.5  # serving GPU must be at least this fraction loaded (weights ~24GB/32GB)

PASS=0
FAIL=0

pass() { echo "PASS: $1"; ((PASS++)); }
fail() { echo "FAIL: $1"; ((FAIL++)); }
info() { echo "      $1"; }

gpu_count() { nvidia-smi -L 2>/dev/null | rg -c 'GPU ' || true; }

gpu_mem_mib() { # $1 = gpu index
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

gpu_total_mib() { # $1 = gpu index
  nvidia-smi -i "$1" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

teardown() {
  docker rm -f vllm >/dev/null 2>&1
  # wait for GPU memory to drain
  for _ in $(seq 1 12); do
    sleep 5
    local busy=0
    for idx in $(nvidia-smi -L 2>/dev/null | rg -o -P 'GPU \K[0-9]+'); do
      (( $(gpu_mem_mib "$idx") > 500 )) && busy=1
    done
    (( busy == 0 )) && return 0
  done
  return 0
}

# start_server ARGS... -> starts detached, returns 0 if script accepted the flags
start_server() {
  local log="$1"; shift
  ( cd "$REPO_DIR" && ./run.sh -d --port "$PORT" "$@" > "$log" 2>&1 )
}

wait_online() { # sets ONLINE=1/0
  # with -d, engine logs live in docker logs, not the run.sh log file
  ONLINE=0
  local waited=0
  while (( waited < STARTUP_TIMEOUT )); do
    if docker logs vllm 2>&1 | grep -q 'Application startup complete'; then
      ONLINE=1; return 0
    fi
    # fail fast if the container exited without reaching startup
    if [[ -z "$(docker ps --filter name=vllm --format '{{.ID}}')" ]]; then
      return 0
    fi
    sleep "$POLL_INTERVAL"; ((waited += POLL_INTERVAL))
  done
}

check_gpu_loaded() { # $1 = gpu index, $2 = label
  local used total
  used=$(gpu_mem_mib "$1"); total=$(gpu_total_mib "$1")
  if [[ -z "$used" || -z "$total" ]]; then
    fail "$2: could not read GPU $1 memory from nvidia-smi"; return 1
  fi
  local thr=$(( total * 10 / 20 ))  # MIN_LOAD_FRACTION as integer math (0.5 -> total/2)
  if (( used > thr )); then
    pass "$2: GPU $1 loaded (${used} MiB / ${total} MiB used)"
    return 0
  else
    fail "$2: GPU $1 NOT loaded (only ${used} MiB / ${total} MiB used, need > ${thr} MiB)"
    return 1
  fi
}

run_test() { # $1 = test name, rest = run.sh args
  local name="$1"; shift
  echo ""
  echo "=== TEST: $name (./run.sh -d --port $PORT $*) ==="
  local log="$LOGDIR/$(echo "$name" | tr ' /' '__').log"

  teardown
  if start_server "$log" "$@"; then
    pass "$name: run.sh accepted flags and started the container"
  else
    fail "$name: run.sh exited non-zero"
    tail -5 "$log" | info; return 1
  fi

  wait_online "$log"
  if (( ONLINE == 1 )); then
    pass "$name: server came online"
  else
    fail "$name: server did not come online within ${STARTUP_TIMEOUT}s"
    tail -10 "$log" | info; teardown; return 1
  fi

  # API smoke check
  if curl -s --max-time 10 "http://localhost:$PORT/v1/models" | rg -q '"object":"list"'; then
    pass "$name: /v1/models responds"
  else
    fail "$name: /v1/models did not respond"
  fi

  # memory assertions are test-specific; caller sets MEMCHECK array of "idx"
  for idx in "${MEMCHECK[@]}"; do
    check_gpu_loaded "$idx" "$name"
  done

  teardown
}

NGPU=$(gpu_count)
echo "bolt run.sh integration tests — $(date)"
echo "Host GPUs detected: $NGPU"
if (( NGPU < 1 )); then
  echo "FAIL: no GPUs detected on this host; cannot run tests"
  exit 1
fi

### Test 1: single GPU 0 (always runs)
MEMCHECK=(0)
run_test "single-gpu-0" --gpu 0

### Test 2: GPU 1 (only if >= 2 GPUs)
if (( NGPU >= 2 )); then
  MEMCHECK=(1)
  run_test "single-gpu-1" --gpu 1

  ### Test 3: GPUs 0,1 (dual exposure)
  MEMCHECK=(0)
  run_test "dual-gpu-0-1" --gpu 0,1
  # informational: with TP=1 only cuda:0 (=GPU 0) should be loaded
  info "note: --gpu 0,1 with TP=1 loads only the primary (GPU 0); GPU 1 usage above is informational"
else
  echo ""
  echo "SKIP: GPU-1 and dual-GPU tests (host has only 1 GPU)"
fi

echo ""
echo "======================================"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "======================================"
(( FAIL == 0 )) || exit 1
exit 0
