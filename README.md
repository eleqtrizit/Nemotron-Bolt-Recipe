# 5090 NVIDIA Nemotron 3.5 Bolt 74B w/ Lilicorr Spec Dec

A Dockerized vLLM server for [`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) — a 74B hybrid Mamba/attention transformer quantized to NVFP4 (~24 GB), served with:

- **b12x** linear + MoE kernels (NVFP4 W4A16 GEMM)
- **FlashInfer** attention
- **dflash speculative decoding** (LiLiCorr draft model, 7 speculative tokens)
- FP8 KV cache
- ultra_v3 reasoning parser + qwen3_coder tool-call parser

Designed to run under [llama-swap](https://github.com/mostlygeek/llama-swap), so `run.sh` stays in the foreground and forwards stdout.

## Requirements

- Docker with NVIDIA Container Toolkit
- A 32 GB GPU (31.36 GiB) — by default the container sees all GPUs; pin the right card with `./run.sh --gpu "device=N"`
- ~25 GB free disk for weights (cached in `~/.cache`)

## Quick start

```bash
./run.sh                        # build image + serve on port 8080 (default), foreground

# Port
./run.sh --port 9000            # serve on host port 9000 instead
./run.sh --port=9000            # same thing, `=` form

# GPUs (no --gpus is passed by default; Docker exposes all GPUs)
./run.sh --gpu "device=1"       # pin to the second GPU (the 32 GB card this config was tuned for)
./run.sh --gpu "device=0"
./run.sh --gpu all

# Detached / restart-on-boot (llama-swap users: prefer the default foreground mode)
./run.sh -d                     # run detached; logs via `docker logs -f vllm`
./run.sh -d --restart always            # survive container crashes and daemon/host boots
./run.sh -d --restart unless-stopped    # survive boots, but stays stopped after `docker stop`

# KV cache dtype (fp8 is the default and the one tuned for this GPU)
./run.sh --kv-cache-dtype fp8   # default, ~2.6 GiB KV for 160k context
./run.sh --kv-cache-dtype bf16  # full precision; needs ~2x KV memory -> lower --max-model-len
./run.sh --kv-cache-dtype nvfp4 # currently blocked upstream for this model (see README troubleshooting / AGENTS.md)
```

The script builds the image (fast after the first time), then runs the container in the **foreground**. vLLM logs stream to stdout, so llama-swap or systemd can capture them.

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

## Benchmarks (RTX 5090, single GPU, concurrency 1)

Measured with the default config (fp8 KV cache + FlashInfer + dflash speculative
decoding). `tok/s` is end-to-end generation speed; `prefill tok/s` is prompt
processing throughput.

| Context | Task | Prompt tok | Gen tok | Prefill tok/s | tok/s |
|---------|------|-----------:|--------:|--------------:|------:|
| 2k | code | 1,640 | 1,676 | 5,463 | 535 |
| 2k | prose | 1,644 | 1,723 | 5,267 | 345 |
| 8k | code | 6,448 | 1,445 | 6,604 | 535 |
| 8k | prose | 6,452 | 972 | 6,612 | 301 |
| 32k | code | 25,681 | 2,048 | 6,974 | 551 |
| 32k | prose | 25,686 | 1,934 | 6,963 | 352 |
| 64k | code | 51,326 | 1,553 | 6,623 | 507 |
| 64k | prose | 51,331 | 1,263 | 6,574 | 294 |
| 128k | code | 102,616 | 2,048 | 5,890 | 526 |
| 128k | prose | 102,620 | 853 | 5,866 | 290 |
| 200k | code | 160,315 | 2,048 | 5,240 | 418 |
| 200k | prose | 160,321 | 1,373 | 5,222 | 263 |

Rough takeaways:
- ~300–550 tok/s generation at short/medium context; dflash speculative decoding
  makes code generation notably faster (~1.5–2x) than prose since code drafts
  accept more often.
- Prefill holds ~5,200–7,000 prompt tok/s up to 128k, degrading mildly at 200k
  (still >5,000).
- Generation speed degrades gracefully with context: only ~20% off at 200k
  versus 2k.

## Architecture

| File | Purpose |
|------|---------|
| `Dockerfile` | Based on `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989`. Installs b12x, downloads the `ultra_v3_reasoning_parser.py` plugin (via python urllib — the base image has no wget/curl), and runs the full `vllm serve` command as CMD. |
| `entrypoint.sh` | Container entrypoint: runs `vllm download-kernels` (needs a GPU, so it can't run at build time), then execs the CMD. |
| `run.sh` | Builds the image, then `exec`s `docker run`. Default: foreground, `--rm`, all GPUs. Flags: `--port` (default 8080), `--gpu`, `-d`, `--restart` (drops `--rm`), `--kv-cache-dtype`. |

### Container settings

- `--gpus` not passed by default (all GPUs visible); pass `--gpu "device=1"` to pin — the pinned GPU then appears as `cuda:0` inside the container
- `--ipc=host` — shared memory for vLLM/torch
- `-p $PORT:8000` — host `$PORT` → vLLM's `0.0.0.0:8000` inside the container
- `-v ~/.cache:/root/.cache` — persists HF weights, compiled kernels, and torch artifacts across restarts
- `--rm` — present in foreground mode, dropped when `--restart` is set (Docker forbids the combination)

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
| Container sees too many/few GPUs | No `--gpus` is passed by default. Check with `docker exec vllm nvidia-smi`; pin with `./run.sh --gpu "device=1"`. |

## llama-swap integration

`run.sh` is llama-swap friendly:

- stays in the foreground and streams logs to stdout
- shuts the container down cleanly on SIGTERM/SIGINT
- `--rm` avoids stale-container name conflicts on restart

Point llama-swap at `./run.sh --port <port>` and probe `http://localhost:<port>/v1/models` for readiness. Give it a generous startup timeout (e.g. `startup_seconds: 2400`) for the first cold start.
