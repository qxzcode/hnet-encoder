"""
Example script taken from hnet-impl's original repo.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nested

from hnet_encoder_impl import HNet

# Create the model.
with torch.device("cuda"):
    model = HNet()
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} parameters")

# Set up the optimizer and LR scheduler.
base_lr = 3e-4
max_steps = 1000
opt = torch.optim.AdamW(
    model.parameters(),
    base_lr,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)
lrs = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda step: (pct := step / max_steps) and (pct * 10 if pct < 0.1 else (1 if pct < 0.9 else (1 - pct) * 10)),
)


# Example dumb task: random number of repeating letters
def generate_random_letters():
    from random import randint
    from string import ascii_lowercase

    return "".join(randint(0, 10) * c for c in ascii_lowercase)


BOS_ID = 254


def tokenize_string(string: str) -> Tensor:
    return (
        F.pad(torch.frombuffer(bytearray(string.encode()), dtype=torch.uint8), (1, 0), value=BOS_ID).int()
        if string
        else torch.tensor([BOS_ID], dtype=torch.int)
    )


def NJT(ls: list[Tensor]):
    return nested.nested_tensor(ls, layout=torch.jagged)


def random_batches():
    while True:
        tokens = [tokenize_string(generate_random_letters()) for _ in range(32)]
        input_ids = [s[:-1] for s in tokens]
        labels = [s[1:] for s in tokens]
        yield NJT(input_ids).cuda(), NJT(labels).long().cuda()


# Training loop
zero = torch.tensor(0.0, device="cuda")
for step, (input_ids, labels) in zip(range(max_steps), random_batches()):
    with torch.autocast("cuda", torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(logits, labels)
    loss.backward()

    opt.step()
    opt.zero_grad()
    lrs.step()

    if step % 10 == 0:
        print(f"{step=}: {loss.item()=:.3f}")
