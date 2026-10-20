# bolt — vLLM NVIDIA Nemotron 3.5 Bolt 74B server

A Dockerized vLLM server for [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) — a 74B hybrid Mamba/attention transformer quantized to NVFP4 (~24 GB), served with:

- **b12x** linear + MoE kernels (NVFP4 W4A16 GEMM)
- **FlashInfer** attention
- **dflash speculative decoding** (LiLiCorr draft model, 7 speculative tokens)
- FP8 KV cache
- ultra_v3 reasoning parser + qwen3_coder tool-call parser

Designed to run under [llama-swap](https://github.com/mostlygeek/llama-swap), so `run.sh` stays in the foreground and forwards stdout.

## Requirements

- Docker with NVIDIA Container Toolkit
- Two GPUs: the server is pinned to **GPU device 1** (the second GPU in `nvidia-smi`), which must have 32 GB VRAM (31.36 GiB)
- ~25 GB free disk for weights (cached in `~/.cache`)

## Quick start

```bash
./run.sh             # build image + serve on host port 8080 (default)
./run.sh --port 9000 # serve on host port 9000 instead
```

The script builds the image (fast after the first time), then runs the container in the **foreground**. vLLM logs stream to stdout, so llama-swap or systemd can capture them.

### Readiness

First startup takes **10–30 minutes**: weights load, then torch.compile, b12x/FlashInfer JIT warmup, and CUDA graph capture run mostly silently (~0% GPU util, EngineCore at ~25–35% CPU — that's normal). Restarts are much faster because compiled artifacts are cached in `~/.cache`.

Check readiness:

```bash
curl http://localhost:8080/v1/models       # ready when this responds
docker logs -f vllm                        # watch for "Application startup complete"
```

### Example request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Architecture

| File | Purpose |
|------|---------|
| `Dockerfile` | Based on `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989`. Installs b12x, downloads the `ultra_v3_reasoning_parser.py` plugin (via python urllib — the base image has no wget/curl), and runs the full `vllm serve` command as CMD. |
| `entrypoint.sh` | Container entrypoint: runs `vllm download-kernels` (needs a GPU, so it can't run at build time), then execs the CMD. |
| `run.sh` | Builds the image, then `exec`s `docker run` in the foreground with `--rm`. Accepts `--port N`. |

### Container settings

- `--gpus '"device=1"'` — pinned to the second GPU; it appears as `cuda:0` inside the container
- `--ipc=host` — shared memory for vLLM/torch
- `-p $PORT:8000` — host `$PORT` → vLLM's `0.0.0.0:8000` inside the container
- `-v ~/.cache:/root/.cache` — persists HF weights, compiled kernels, and torch artifacts across restarts

## Server configuration that matters

- `--max-model-len 163840` (160k context) + `--gpu-memory-utilization 0.96`
  - The original 262144 context needed 2.99 GiB of KV cache, but the 32 GB card only has ~2.6 GiB free after the 24 GB weights + CUDA graphs. 262144 does **not** fit on one GPU. If you need more context, use `--tensor-parallel-size 2` across both GPUs.
- `--kv-cache-dtype fp8` — halves KV memory vs bf16; may cause slight accuracy loss
- `--max-num-seqs 1` — single concurrent request
- `--speculative-config` — dflash with the LiLiCorr draft model, 7 speculative tokens

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Free memory on device ... less than desired GPU memory utilization` | Another process is using GPU 1. Stop it or lower `--gpu-memory-utilization` in the Dockerfile (0.96 is the known-good value; don't raise it to 0.98). |
| `To serve at least one request ... (X GiB KV cache is needed)` | `--max-model-len` is too large for available KV memory. See the config note above. |
| `[ERROR] vision_final_layernorm ... not documented` | Harmless transformers doc-lint warning, not a real error. |
| Startup seems stuck, 0% GPU, no new logs | Normal — it's compiling (torch.compile / JIT / CUDA graph capture). Wait for `Application startup complete`. |
| Container died and `docker logs` says "No such container" | The container runs with `--rm`, so it's removed on exit. Capture logs when running: `./run.sh > vllm.log 2>&1`. |
| GPU 0 is being used instead of GPU 1 | Verify with `docker exec vllm nvidia-smi` — you should see exactly one GPU. |

## llama-swap integration

`run.sh` is llama-swap friendly:

- stays in the foreground and streams logs to stdout
- shuts the container down cleanly on SIGTERM/SIGINT
- `--rm` avoids stale-container name conflicts on restart

Point llama-swap at `./run.sh --port <port>` and probe `http://localhost:<port>/v1/models` for readiness. Give it a generous startup timeout (e.g. `startup_seconds: 2400`) for the first cold start.
