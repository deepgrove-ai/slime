```bash

# Download model
hf download Qwen/Qwen3-0.6B --local-dir /root/Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download Qwen/Qwen3-14B --local-dir /root/Qwen3-14B
hf download RedMod/qsft_14b --local-dir /root/qsft_14b
hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B
# Download training dataset (dapo-math-17k)
python3 VeOmni/scripts/moe_ckpt_merge/moe_merge.py --raw_hf_path /root/Qwen3-30B-A3B  --merge_hf_path /root/Qwen3-30B-A3B-merge
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/aime-2024


# Prepare for megatron
cd /root/slime
source scripts/models/qwen3-0.6B.sh
PYTHONPATH=/root/Megatron-LM uv run --no-sync python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-0.6B \
    --save /root/Qwen3-0.6B_torch_dist

source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM uv run --no-sync python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```
