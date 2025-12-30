from collections.abc import Iterable, Iterator
from typing import TypedDict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, nested
from tqdm.auto import tqdm

from hnet_encoder_impl import HNet


# Example dumb task: random number of repeating letters
def generate_random_letters():
    from random import randint
    from string import ascii_lowercase

    return "".join(randint(0, 10) * c for c in ascii_lowercase)


def tokenize_string(string: str) -> Tensor:
    BOS_ID = 254
    return (
        F.pad(torch.frombuffer(bytearray(string.encode()), dtype=torch.uint8), (1, 0), value=BOS_ID).int()
        if string
        else torch.tensor([BOS_ID], dtype=torch.int)
    )


def NJT(ls: list[Tensor]):
    return nested.nested_tensor(ls, layout=torch.jagged)


class DatasetItem(TypedDict):
    """An item from the `allenai/c4` dataset."""

    text: str
    timestamp: str
    url: str


class Sampler:
    def __init__(self, dataset: Iterator[DatasetItem], seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self._buffer = b""

    def sample(self) -> bytes:
        while len(self._buffer) < self.seq_len:
            self._buffer += next(self.dataset)["text"].encode("utf-8") + b"\n"

        result = self._buffer[: self.seq_len]
        self._buffer = self._buffer[self.seq_len :]
        return result


class BatchSampler:
    def __init__(self, dataset: Iterable[DatasetItem], seq_len: int, batch_size: int):
        data_iter = iter(dataset)
        self._samplers = [Sampler(data_iter, seq_len) for _ in range(batch_size)]

    def sample(self) -> list[bytes]:
        return [s.sample() for s in self._samplers]


def random_batches(batch_sampler: BatchSampler):
    while True:
        token_ids = torch.stack(
            [torch.frombuffer(bytearray(seq), dtype=torch.uint8) for seq in batch_sampler.sample()]
        ).to(device="cuda", dtype=torch.long)
        mask = torch.rand_like(token_ids, dtype=torch.float) < 0.15
        yield token_ids, mask


MASK_ID = 255


def main():
    # Create the model.
    print("Creating model...")
    with torch.device("cuda"):
        model = HNet()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"    Model has {num_params:,} parameters")

    # Set up the optimizer and LR scheduler.
    print("Creating optimizer and LR scheduler...")
    base_lr = 3e-4
    max_steps = 100_000
    optimizer = torch.optim.AdamW(
        model.parameters(),
        base_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (pct := step / max_steps) and (pct * 10 if pct < 0.1 else (1 if pct < 0.9 else (1 - pct) * 10)),
    )

    # Initialize the dataset and sampler.
    print("Initializing dataset and sampler...")
    dataset_train = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(seed=0)
    batch_sampler = BatchSampler(dataset_train, seq_len=512, batch_size=32)

    # Training loop
    print("Starting training loop\n")
    try:
        for step, (token_ids, mask) in zip(tqdm(range(max_steps), unit="step"), random_batches(batch_sampler)):
            input_ids = torch.where(mask, MASK_ID, token_ids)
            target_ids = torch.where(mask, token_ids, -100)

            # with torch.autocast("cuda", torch.bfloat16):
            if True:
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % 1 == 0:
                print(f"{step=}: {loss.item()=:.3f}")
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(model, "model.pt")
        print("Saved model.pt")


if __name__ == "__main__":
    main()
