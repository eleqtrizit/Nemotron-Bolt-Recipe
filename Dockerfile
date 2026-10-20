FROM vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989

RUN pip install --no-cache-dir b12x

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN python3 -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/resolve/main/ultra_v3_reasoning_parser.py', '/ultra_v3_reasoning_parser.py')"

ENV VLLM_USE_V2_MODEL_RUNNER=1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["vllm", "serve", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--trust-remote-code", \
    "--load-format", "auto", \
    "--tensor-parallel-size", "1", \
    "--max-model-len", "163840", \
    "--max-num-seqs", "1", \
    "--enable-chunked-prefill", \
    "--gpu-memory-utilization", "0.96", \
    "--kv-cache-dtype", "fp8", \
    "--linear-backend", "b12x", \
    "--moe-backend", "b12x", \
    "--attention-backend", "FLASHINFER", \
    "--mamba-backend", "flashinfer", \
    "--mamba-ssm-cache-dtype", "bfloat16", \
    "--mamba-cache-dtype", "bfloat16", \
    "--mamba-cache-mode", "align", \
    "--seed", "42", \
    "--allowed-local-media-path", "/", \
    "--limit-mm-per-prompt", "{\"video\":1,\"image\":1}", \
    "--video-pruning-rate", "0.5", \
    "--media-io-kwargs", "{\"video\":{\"fps\":2,\"num_frames\":256}}", \
    "--enable-auto-tool-choice", \
    "--tool-call-parser", "qwen3_coder", \
    "--reasoning-parser-plugin", "/ultra_v3_reasoning_parser.py", \
    "--reasoning-parser", "ultra_v3", \
    "--speculative-config", "{\"method\":\"dflash\",\"model\":\"nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr\",\"num_speculative_tokens\":7,\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASHINFER\",\"draft_load_config\":{\"load_format\":\"auto\"}}"]
