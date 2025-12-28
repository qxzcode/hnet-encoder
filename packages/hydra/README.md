# Hydra sequence mixer

Adapted from https://github.com/goombalab/hydra.

## Basic usage

```python
from hydra import Hydra

device = torch.device("cuda")

hydra = Hydra(512, learnable_init_states=True, bias=True)
hydra.to(device)

x = torch.randn(1, 9001, 512, device=device)
y = hydra(x)
```
