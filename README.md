# H-Net Encoder

## Quickstart

```bash
uv sync --all-packages
export MLFLOW_TRACKING_URI="http://localhost:6767"
CUDA_VISIBLE_DEVICES=6,7 uv run torchrun --nproc-per-node 1 --standalone example.py # To train
```
