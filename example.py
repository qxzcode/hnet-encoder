import os
from collections.abc import Generator
from typing import Any

import torch
from datasets import load_dataset
from torch import Tensor as TT
from torch import nested
from torch.distributed import device_mesh as tdm
from torch.distributed import fsdp

from hnet_impl import ByteTokenizer, HNetConfig, HNetLM, completion_sync


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

def main() -> None:
    ## dist init
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    mesh = tdm.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

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
    base_lr, max_steps = 3e-4, 1000
    opt = torch.optim.AdamW(
        [dict(params=ls, lr=base_lr * lr_mod) for ls, lr_mod in zip(m.split_params_by_hierachy(), c.lambda_s())],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    lrs = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: (pct := step / max_steps) and (pct * 10 if pct < 0.1 else (1 if pct < 0.9 else (1 - pct) * 10)),
    )


    max_seq_len = 512 # 1024 causes OOM
    log_incr = 10
    max_new = max_steps // log_incr - 1
    with m.sampling_mode():
        print(completion_sync("", t, m, max_new=max_new))

    # Load dataset
    ds = load_dataset("allenai/c4", "en", streaming=True)
    ds_train = ds["train"]
    ds_train = ds_train.shuffle(0)
    ds_train_iter = iter(ds_train)

    ## training loop
    zero = torch.tensor(0.0, device="cuda")
    for step, (iids, lbls) in zip(range(max_steps), random_batches(t, ds_train_iter, max_seq_len)):
        with torch.autocast("cuda", torch.bfloat16):
            (l_avg, l_sum), extra = m(iids.cuda(), lbls.cuda())
            l_ratio = sum([e.loss_ratio for e in extra], zero)
            loss = l_avg + l_ratio
        loss.backward()

        opt.step()
        opt.zero_grad()
        lrs.step()

        if step % log_incr == 0:
            print(f"{step=}: {l_avg.item()=:.3f} {l_ratio.item()=:.3f}")

    with m.sampling_mode():
        print(completion_sync("", t, m, max_new=max_new))

if __name__ == "__main__":
    main()
