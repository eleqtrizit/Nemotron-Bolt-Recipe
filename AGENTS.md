# bolt — vLLM Nemotron Bolt 74B server

Dockerized vLLM serving `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` with b12x
linear/MoE kernels, FlashInfer attention, and dflash speculative decoding.
Intended to run under llama-swap (therefore `run.sh` stays in the foreground).

## Files

- `Dockerfile` — based on `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989`.
  Bakes in `pip install b12x`, the `ultra_v3_reasoning_parser.py` plugin (downloaded
  via python urllib — the image has no wget/curl), and the full `vllm serve` command
  as CMD. Entrypoint is `entrypoint.sh`, which runs `vllm download-kernels` at
  container start (it needs a GPU, so it cannot run during `docker build`).
- `entrypoint.sh` — runs `vllm download-kernels`, then execs the CMD.
- `run.sh` — builds the image, then runs the container (foreground by default).
  Flags: `--port` (default 8080), `-d`, `--restart`, `--gpu`, `--tp`, `--kv-cache-dtype`.
Tensor parallelism defaults to 1. Auto-TP=2 was tried and reverted (2026-10-20):
with `--tensor-parallel-size 2` the engine fails at model load with
`ValueError: IQ2_XXS dense weights require BF16, K256 and N8` — the mixed-
precision checkpoint's dense/shared-expert weights cannot be sharded. TP=2 stays
available via `--tp 2` for future checkpoint/kernel updates. The TP value
reaches the server via the `TP` env var, rewritten into the CMD by
entrypoint.sh (same mechanism as KV_CACHE_DTYPE). Also note TP=2 needs both
GPUs free: GPU 0 with >2 GiB already used trips the gpu-memory-utilization check.

## Usage

```bash
./run.sh             # serve on host port 8080 (default)
./run.sh --port 9000 # serve on host port 9000
./run.sh -d --restart always           # detached + restart on container exit and daemon boot
./run.sh --gpu "device=1"              # pin to the second GPU only
```

Flags: `--port N`, `-d/--detach`, `--restart [always|unless-stopped|on-failure]`
(default `unless-stopped`; drops `--rm` since Docker forbids the combination),
`--kv-cache-dtype fp8|bf16|nvfp4` (fp8 default).

Logs stream to stdout (llama-swap captures them). On SIGTERM/SIGINT the container
shuts down cleanly; `--rm` auto-removes it so restarts never hit name conflicts.

Easy checks:

```bash
curl http://localhost:8080/v1/models   # ready when this responds
docker exec vllm nvidia-smi            # container sees exactly one GPU (device 1)
```

## Key facts / gotchas

- **GPU:** by default `run.sh` pins the container to the GPU with the most
  free memory (`nvidia-smi` query) — docker run without `--gpus` exposes NO
  GPUs, and `--gpus all` makes a busy cuda:0 the primary (GPU 0 hosts other
  services on this machine). Override with `--gpu "device=1"`. Historically
  the serving card is the second GPU (32 GB / 31.36 GiB, `cuda:0` inside the
  container); the KV cache math below assumes that card.
- **TP=2 does not work with this checkpoint:** model load fails with
  `IQ2_XXS dense weights require BF16, K256 and N8` (mixed-precision dense
  weights can't be sharded). `--tp 2` remains opt-in for future tests.
- **Context length math:** the ~24GB quantized weights + CUDA graphs leave only
  ~2.6 GiB for KV cache. The original 262144 context needed 2.99 GiB and failed;
  vLLM's own estimate of the max was 175,344. Current values that work:
  `--max-model-len 163840` (160k) + `--gpu-memory-utilization 0.96`.
  Do not raise max-model-len back to 262144 on one 32 GB GPU. Alternatives if
  more context is needed: `--tensor-parallel-size 2` across both GPUs (run.sh
does this automatically when it sees 2+ GPUs and no `--gpu` pin).
- **0.98 gpu-memory-utilization fails**: the card only has 30.37 GiB free at
  startup; vLLM refuses requests above that. 0.96 is the working value.
- **First startup is slow (10–30 min):** weights load, then torch.compile,
  b12x/FlashInfer JIT warmup, and CUDA graph capture run mostly silently with
  ~0% GPU util and the EngineCore process at 25–35% CPU. That's normal; wait for
  `Application startup complete` in the logs. `~/.cache` is mounted to
  `/root/.cache`, so compiled artifacts and HF weights persist and restarts are
  much faster.
- **Run from a shell you control:** `nohup ./run.sh &` from a sandboxed/short-lived
  shell will kill the container when the shell dies. For long-lived runs use a
  real terminal, systemd, or llama-swap itself.
- **Debugging:** in the default foreground mode the container is removed on exit
  (`--rm`), so `docker logs` is unavailable after a crash. Capture output instead:
  `./run.sh > vllm.log 2>&1` and inspect the file. With `-d` (or `--restart`),
  `docker logs vllm` works.
- Host port is mapped `$PORT:8000`; vLLM binds `0.0.0.0:8000` inside the container
  (explicit `--host 0.0.0.0 --port 8000` in the Dockerfile CMD).
- **KV cache dtype:** fp8 by default; selectable at runtime via
  `run.sh --kv-cache-dtype fp8|bf16|nvfp4` (entrypoint rewrites the CMD flag from
  the `KV_CACHE_DTYPE` env var). bf16 doubles KV memory — drop `--max-model-len`
  accordingly. See the NVFP4 section for why nvfp4 is currently blocked.

## Startup-phase log lines worth knowing

- `Available KV cache memory: X GiB` — if startup fails with
  `To serve at least one request ... (X GiB KV cache is needed)`, the context
  length is too large for available memory (see gotcha above).
- `ValueError: Free memory on device cuda:0 ... less than desired GPU memory
  utilization` — some other process is using the GPU; lower `gpu-memory-utilization`.
- Docs/config lint lines like `[ERROR] vision_final_layernorm ... not documented`
  are harmless transformers warnings, not real errors.
- Triton cubin reload warnings ("Cubin file saved by TritonBundler not found")
  used to appear on every restart because torch inductor cached to the
  container's ephemeral /tmp. Fixed permanently via
  `TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor` and
  `TRITON_CACHE_DIR=/root/.cache/triton` env vars in the Dockerfile (both land
  on the persistent ~/.cache mount). If they ever reappear, check that these
  env vars are still set.

## NVFP4 KV cache — BLOCKED upstream (tested 2026-10-20)

`--kv-cache-dtype nvfp4` (halves KV memory vs fp8; would likely restore the full
262144 context on this 32 GB GPU) does NOT work for this model/GPU today. The
flag is accepted by vLLM but backend selection refuses it. Tested exhaustively
on both the nightly snapshot (77154ca6) and the real v0.32.0 release image,
every backend:

- FLASHINFER: "the sink prefill wrapper cannot read an NVFP4 KV cache" <-- our blocker
- FLASH_ATTN: needs FA3 on SM90 or FA4 on SM100, not SM120
- TRITON_ATTN / TRITON_FLASH_ATTN / TRITON_FLASHINFER: head size / CC not supported
- B12X: fp8/fp8_e4m3 only by design

Our model registers attention sinks, and FlashInfer's fa2 prefill wrapper
cannot read attention sinks over an NVFP4 (block-scaled) KV cache. vLLM PR
#46963 ("NVFP4 KV cache on SM8x/SM12x") deliberately kept this refusal. Not
fixable by config; blocked on upstream kernels.

## PRs to watch (retest when these land)

- https://github.com/flashinfer-ai/flashinfer/pull/6021 — attention sinks over
  NVFP4 KV cache in prefill and decode (THE blocker for us)
- https://github.com/flashinfer-ai/flashinfer/pull/6024 — NVFP4 scale-factor
  tile load fix (head_size 64 path, merged alongside 6021)
- After both ship in a flashinfer-python release, vLLM must bump its pin
  (currently flashinfer 0.7.0.post1). Then retest with the v0.32.0+ image:
  `./run.sh --kv-cache-dtype nvfp4`. Expected ~3.3x KV capacity (PR #46963
  measured 1.45M -> 4.8M tokens on SM80).

## Repro of the failure (for future agents)

1. Set `--kv-cache-dtype nvfp4` in the Dockerfile.
2. `./run.sh` -> container exits after a few minutes with:
   `ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not valid
   for this configuration. Reason: ['the sink prefill wrapper cannot read an
   NVFP4 KV cache']`
3. Do not burn more time trying other attention backends; the matrix above is
   exhaustive as of v0.32.0.
