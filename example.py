import os
from collections.abc import Generator
from typing import Any

import mlflow
import torch
from datasets import load_dataset
from pydantic import BaseModel
from torch import Tensor as TT
from torch import nested
from torch.distributed import device_mesh as tdm
from torch.distributed import fsdp
from utils import parse_cfg

from hnet_impl import ByteTokenizer, HNetConfig, HNetLM


class Args(BaseModel):
    # Metadata
    exp_name: str = ""
    """If this is not empty, MLFlow will track this run under this name."""
    
    # Experiment
    max_seq_len: int = 512
    """Training examples will be truncated to this size, measured in bytes."""
    ds_seed: int = 100
    """Seed to use when shuffling the dataset."""
    train_steps: int = 1000
    """Number of training iterations."""
    
    # Optimizer
    base_lr: float = 3e-4
    aw_b1: float = 0.9
    """AdamW beta1."""
    aw_b2: float = 0.95
    """AdamW beta2."""
    aw_weight_decay: float = 0.01


def main() -> None:
    ## dist init
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    mesh = tdm.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    # Init experiment
    args = parse_cfg(Args)
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment("hnet_encoder")
    with mlflow.start_run(run_name=args.exp_name or None):
        mlflow.log_params(args.model_dump())

        ## create model
        t = ByteTokenizer()
        c = HNetConfig.create_reasonable_config(D=[512, 1024], arch=["m4", "T9"])
        with torch.device("cuda"):
            m = HNetLM(c)

        ## fsdp/compile
        m.backbone.block_compile(ac=False)
        m.apply_fsdp(  # default: BF16, ZeRO2, 1D mesh
            m,
            mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            reshard_after_forward=False,
            mesh=mesh["dp"],
        ) if world_size > 1 else m

        ## optim / lr sched
        opt = torch.optim.AdamW(
            [dict(params=ls, lr=args.base_lr * lr_mod) for ls, lr_mod in zip(m.split_params_by_hierachy(), c.lambda_s())],
            betas=(args.aw_b1, args.aw_b2),
            weight_decay=args.aw_weight_decay,
        )
        lrs = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: (pct := step / args.train_steps) and (pct * 10 if pct < 0.1 else (1 if pct < 0.9 else (1 - pct) * 10)),
        )

        # Load dataset
        ds = load_dataset("allenai/c4", "en", streaming=True)
        ds_train = ds["train"]
        ds_train = ds_train.shuffle(seed=args.ds_seed)
        ds_train_iter = iter(ds_train)

        ## training loop
        zero = torch.tensor(0.0, device="cuda")
        for step, (iids, lbls) in zip(range(args.train_steps), random_batches(t, ds_train_iter, args.max_seq_len)):
            with torch.autocast("cuda", torch.bfloat16):
                (l_avg, l_sum), extra = m(iids.cuda(), lbls.cuda())
                l_ratio = sum([e.loss_ratio for e in extra], zero)
                loss = l_avg + l_ratio
            loss.backward()

            opt.step()
            opt.zero_grad()
            lrs.step()

            # Log step
            log_dict = {
                "l_avg": l_avg.cpu().item(),
                "l_ratio": l_ratio.cpu().item(),
                "loss": loss.cpu().item(),
            }
            for lr_idx, lr in enumerate(lrs.get_last_lr()):
                log_dict[f"lr{lr_idx}"] = lr
            mlflow.log_metrics(log_dict, step)

def get_one_example(ds_iter: Generator, max_seq_len: int) -> str:
    return next(ds_iter)["text"][:max_seq_len]


def NJT(ls: list[TT]) -> TT:
    return nested.nested_tensor(ls, layout=torch.jagged)


def random_batches(t: ByteTokenizer, ds_iter: Generator, max_seq_len: int) -> Generator[tuple[TT, TT], Any, None]:
    while True:
        tokens = t.encode([get_one_example(ds_iter, max_seq_len) for _ in range(32)])
        iids = [s[:-1] for s in tokens]
        lbls = [s[1:] for s in tokens]
        yield NJT(iids), NJT(lbls).long()

if __name__ == "__main__":
    main()
