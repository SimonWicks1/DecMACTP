import torch as th
import torch.nn as nn
import math


@th.no_grad()
def sparse_init(module, sparsity_factor=0.9, **kwargs):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.xavier_normal_(module.weight)
        rows, cols = module.weight.shape
        num_zeros = int(math.ceil(sparsity_factor * cols))
        for i in range(rows):
            row_perm = th.randperm(cols)
            zero_indices = row_perm[:num_zeros]
            module.weight.data[i, zero_indices] = 0.0
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


@th.no_grad()
def lecun_init(module, **kwargs):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(module.weight, mean=0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@th.no_grad()
def xavier_init(module, gain=1.0, **kwargs):
    if hasattr(module, "weight") and module.weight is not None:
        th.nn.init.xavier_uniform_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            th.nn.init.zeros_(module.bias)


@th.no_grad()
def ibarz2022init(module, key, graph_spec, **kwargs):
    if graph_spec[key][2] == "scalar":
        xavier_init(module, **kwargs)
    else:
        lecun_init(module, **kwargs)


_WEIGHTS_INIT_FUNCS = {
    "sparse": sparse_init,
    "lecun": lecun_init,
    "xavier": xavier_init,
    "ibarz2022": ibarz2022init,
}


def multipurpose_init(module, gnn_init: str, encoder_init: str, **kwargs):
    if "key" in kwargs and encoder_init is not None:
        return _WEIGHTS_INIT_FUNCS[encoder_init](module, **kwargs)
    elif gnn_init is not None:
        return _WEIGHTS_INIT_FUNCS[gnn_init](module, **kwargs)
