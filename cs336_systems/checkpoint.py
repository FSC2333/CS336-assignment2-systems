import gc
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock


# xl config
N = 32
batch_size = 4
d_model = 2560
d_ff = 10240
num_heads = 16
context_length = 2048
device = "cuda"


def make_block():
    return TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        positional_encoder=RotaryEmbedding(
            dim=d_model // num_heads,
            context_length=context_length,
        ),
    )


# 真实 32 层；如果显存不够，可以先用 shared block 做 activation-only sanity check，
# 但正式 profile 最好用 32 个独立 block。
blocks = nn.ModuleList([make_block().to(device) for _ in range(N)])

# 尽量使用 compile；也可以先关掉 compile 做 baseline。
blocks = nn.ModuleList([
    torch.compile(block, fullgraph=True)
    for block in blocks
])


def run_segment(blocks, start, end, x):
    for i in range(start, end):
        x = blocks[i](x)
    return x


def forward_one_level_checkpoint(blocks, x, chunk_size: int):
    for start in range(0, len(blocks), chunk_size):
        end = min(start + chunk_size, len(blocks))

        # 注意用默认参数绑定 start/end，避免 Python closure 捕获最后一个值
        x = checkpoint(
            lambda z, start=start, end=end: run_segment(blocks, start, end, z),
            x,
            use_reentrant=False,
        )
    return x


def profile_chunk_size(chunk_size: int):
    for p in blocks.parameters():
        p.grad = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    x = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=device,
        requires_grad=True,
    )

    y = forward_one_level_checkpoint(blocks, x, chunk_size)
    loss = y.sum()
    loss.backward()

    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # clean up before next measurement
    del x, y, loss
    for p in blocks.parameters():
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()

    return peak_gib


# CUDA / compile warmup，避免第一次 CUDA 初始化落在 checkpoint 内部
torch.cuda.init()
_ = torch.empty((), device=device)

# 建议比较 sqrt(N) 附近；如果长上下文 attention residual 很大，再加 1/2
chunk_sizes = [1, 2, 4, 6, 8, 16, 32]

results = {}
for k in chunk_sizes:
    try:
        peak = profile_chunk_size(k)
        results[k] = peak
        print(f"chunk_size={k:>2}: peak={peak:.2f} GiB")
    except torch.cuda.OutOfMemoryError:
        results[k] = float("inf")
        print(f"chunk_size={k:>2}: OOM")
        torch.cuda.empty_cache()

best_k = min(results, key=results.get)
print(f"\nBest chunk_size = {best_k}, peak = {results[best_k]:.2f} GiB")