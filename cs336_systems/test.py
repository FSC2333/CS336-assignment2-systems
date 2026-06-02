# import torch
# from torch import nn

# x = torch.randn((4, 512, 2560), requires_grad=True)

# class RMSNorm(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         eps: float = 1e-5,
#         device=None,
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
#         self.eps = eps

#     def forward(self, x):
#         rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         x = x * rms
#         return self.weight * x

# def pack_hook(t):
#     shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
#     print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
#     return t

import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import RotaryEmbedding, TransformerBlock

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
device = "cuda"

torch.cuda.init()
_ = torch.empty((), device=device)

raw_block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads,
        context_length=context_length,
    ),
).to(device)

block = torch.compile(raw_block, fullgraph=True)

x = torch.randn(
    (4, context_length, d_model),
    device=device,
    requires_grad=True,
)

# warmup：让 torch.compile / CUDA / Triton 初始化发生在 checkpoint 外面
x_warmup = torch.randn_like(x, requires_grad=True)
y_warmup = block(x_warmup)
y_warmup.sum().backward()

raw_block.zero_grad(set_to_none=True)
x.grad = None
torch.cuda.synchronize()

total_size_bytes = 0

# 更稳妥地跳过参数和参数 view
param_storage_ptrs = {
    p.untyped_storage().data_ptr()
    for p in raw_block.parameters()
}

def is_param_storage(t):
    return t.untyped_storage().data_ptr() in param_storage_ptrs

def pack_hook(t):
    global total_size_bytes

    if is_param_storage(t):
        return t

    total_size_bytes += t.numel() * t.element_size()
    print(
        f"Saving residual: shape={t.shape}, "
        f"dtype={t.dtype}, device={t.device}, grad_fn={t.grad_fn}"
    )
    return t

def unpack_hook(t):
    print(
        f"Loading residual: shape={t.shape}, "
        f"dtype={t.dtype}, device={t.device}, grad_fn={t.grad_fn}"
    )
    return t

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
    # 如果只想统计 checkpointed forward 保存了多少，可以不 backward
    # 如果想观察 checkpoint backward 重算，就取消下一行注释：
    # y.sum().backward()

print(
    f"Total size of saved tensors during checkpointed forward: "
    f"{total_size_bytes / (1024**2):.2f} MiB"
)