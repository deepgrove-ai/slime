```bash

# Download model
hf download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B

# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024


# Prepare for megatron
cd /root/slime
source scripts/models/qwen3-0.6B.sh
PYTHONPATH=/root/Megatron-LM uv run python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-0.6B \
    --save /root/Qwen3-0.6B_torch_dist
```