# H-Net Encoder

## Quickstart

```bash
uv sync --all-packages
CUDA_VISIBLE_DEVICES=6,7 uv run torchrun --nproc-per-node 2 --standalone example.py # To train
```

Note: The example script will show no progress until the very last 3 seconds, so don't worry if it "hangs". It ended up
taking 5-10 minutes for me.
