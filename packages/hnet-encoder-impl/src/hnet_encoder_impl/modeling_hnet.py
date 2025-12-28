import torch
import torch.nn.functional as F
from torch import Tensor, nn

### ################
### H-Net submodules
### ################


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


# NOTE: it's possible to fuse q/k/res proj into a single gemm kernel, but only iff they are of equal precision.
class RoutingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.q_proj_layer = Lin(d, d)
        self.k_proj_layer = Lin(d, d)
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py#L49
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d))
            self.k_proj_layer.weight.copy_(torch.eye(d))

    def forward(self, r_flat: Tensor, r_cu: Tensor):
        k_flat = self.k_proj_layer(r_flat)
        q_flat = QProjPadded.apply(r_flat, self.q_proj_layer.weight, k_flat, r_cu)
        cos_sim = F.cosine_similarity(q_flat, k_flat, dim=-1)
        p_flat = (0.5 - cos_sim / 2).clamp(0.0, 1.0)
        b_flat = p_flat >= 0.5
        p_select_cu = F.pad(b_flat.cumsum(0), (1, 0))[r_cu]
        return p_flat, b_flat, p_select_cu


class DeChunkLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # for EMA scan kernel.
        self.block_size = 256
        self.headdim = 32
        self.nheads, _r = divmod(d, self.headdim)
        assert _r == 0
        A = -torch.ones(self.nheads, device="cuda", dtype=torch.float32)
        self.register_buffer("A", A, persistent=False)

    @staticmethod
    def forward_flat(
        h_flat: Tensor,
        b_flat: Tensor,
        p_selected_flat: Tensor,
        h_seq_idx: Tensor,
        *,
        eps=1e-4,
        nheads: int,
        headdim: int,
        block_size: int,
        A: Tensor,
    ):
        p = p_selected_flat.float().clamp(eps, 1 - eps)

        dt = -torch.log1p(-p.float())[..., None]
        h = (h_flat.float() / dt).type_as(h_flat)
        c = torch.ones_like(p := p.type_as(h)[None, :, None, None])

        z_bar_flat = mamba_chunk_scan_combined(
            h.view(1, -1, nheads, headdim),
            dt.expand(-1, nheads).to(h.dtype)[None],
            A,
            p,
            c,
            chunk_size=block_size,
            seq_idx=h_seq_idx,
        )[0].view(-1, h.shape[-1])

        inner2outer_idx = b_flat.cumsum(0) - 1
        return z_bar_flat.index_select(0, inner2outer_idx)

    def forward(self, h_flat: Tensor, b_flat: Tensor, p_selected_flat: Tensor, h_seq_idx: Tensor, *, eps=1e-4):
        return self.forward_flat(
            h_flat,
            b_flat,
            p_selected_flat,
            h_seq_idx,
            eps=eps,
            nheads=self.nheads,
            headdim=self.headdim,
            block_size=self.block_size,
            A=self.A,
        )


### #################
### Final HNet Module
### #################


class HNet(nn.Module):
    def __init__(self):
        super().__init__()

        ...
