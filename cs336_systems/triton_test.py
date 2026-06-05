from __future__ import annotations

import argparse

import triton
import triton.language as tl
import torch


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # 输入指针
    output_ptr,         # 输出指针
    x_stride_row, x_stride_dim,  # stride 告诉我们如何沿 tensor 每个轴移动一个元素
    weight_stride_dim,           # 通常是 1
    output_stride_row,           # 通常是 1
    NUM_ROWS,
    D: tl.constexpr,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,   # tile 形状必须在编译时已知
):
    # 每个 instance 会计算 x 中一个行 tile 的 weighted sum。
    # tl.program_id 让我们知道当前正在运行哪个 thread block。
    row_tile_idx = tl.program_id(0)

    # Block pointer 让我们可以从一段 ND 内存区域中选择数据，
    # 并移动这个选择区域。
    # block pointer 必须知道：
    # - 指向 tensor 第一个元素的指针
    # - tensor 的整体形状，用于处理越界访问
    # - 每个维度的 stride，用于正确处理内存布局
    # - 起始 block 的 ND 坐标，也就是 offsets
    # - 每次 load/store 的 block shape
    # - 维度在内存中从 major 到 minor 的顺序
    #   axes = np.argsort(strides)，用于优化；
    #   在 Hopper 及之后架构上支持 TMA 时需要
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 初始化一个用于写入的 buffer
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前 block pointer 指向的数据。
        # 因为 ROWS_TILE_SIZE 不一定整除 NUM_ROWS，
        # D_TILE_SIZE 也不一定整除 D，
        # 所以两个维度都需要 boundary checks。
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)

        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )  # (D_TILE_SIZE,)

        # 计算当前行 tile 的 weighted sum。
        output += tl.sum(row * weight[None, :], axis=1)

        # 将 pointer 移动到下一个 tile。
        # 这些是以 rows, columns 表示的坐标增量。
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # 每一行对应一个标量。
    # 因为 ROWS_TILE_SIZE 不一定整除 NUM_ROWS，所以需要 boundary check。
    tl.store(output_block_ptr, output, boundary_check=(0,))



@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,          # 输入
    grad_output_ptr,            # 输入梯度
    grad_x_ptr, partial_grad_weight_ptr,  # 输出梯度
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS,
    D: tl.constexpr,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )  # (ROWS_TILE_SIZE,)

        # grad_x 的外积
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )  # (D_TILE_SIZE,)

        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 为 grad_weight 尽可能在当前 row tile 内做 reduction
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)

        grad_weight_row = tl.sum(
            row * grad_output[:, None],
            axis=0,
        )

        tl.store(
            partial_grad_weight_block_ptr,
            grad_weight_row[None, :],
            boundary_check=(0, 1),
        )

        # 沿 D 维度移动到下一个 tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 缓存 x 和 weight，供 backward 使用。
        # backward 时我们只会收到相对于输出 tensor 的梯度，
        # 需要计算相对于 x 和 weight 的梯度。
        D = x.shape[-1]
        output_dims = x.shape[:-1]

        # 将输入 tensor reshape 成 2D。
        input_shape = x.shape
        x = x.reshape(-1, D).contiguous()

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.dtype == weight.dtype, "Expected x and weight to have the same dtype"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # 大约让 embedding 维度循环 16 次
        ctx.D_TILE_SIZE = max(1, triton.next_power_of_2(D) // 16)

        # 每个 thread 处理 16 个 batch element
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        # 需要初始化空结果 tensor。
        # 注意这些元素不一定是 0！
        y = torch.empty(output_dims, device=x.device, dtype=x.dtype)

        # 用 1D grid 启动 n 个 kernel instance。
        n_rows = y.numel()

        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous().view(-1)

        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE = ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # 我们的策略是：
        # 每个 thread block 先写入一个 partial buffer，
        # 然后对这个 buffer 做 reduction，得到最终梯度。
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device,
            dtype=x.dtype,
        )

        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x.view(ctx.input_shape), grad_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small smoke test for the Triton weighted-sum autograd function.")
    parser.add_argument("--num-rows", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This Triton test requires CUDA.")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(args.dtype)

    shape = (args.num_rows, args.d_model)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn((args.d_model,), device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    f_weightedsum = WeightedSumFunc.apply
    y = f_weightedsum(x, weight)
    y_ref = (x_ref * weight_ref).sum(dim=-1)

    atol = 1e-4 if dtype == torch.float32 else 5e-2
    rtol = 1e-4 if dtype == torch.float32 else 5e-2
    print(y)
    print(f"forward max_abs_diff={(y - y_ref).abs().max().item():.6e}")
    print(f"forward allclose={torch.allclose(y, y_ref, atol=atol, rtol=rtol)}")

    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    assert x.grad is not None
    assert weight.grad is not None
    assert x_ref.grad is not None
    assert weight_ref.grad is not None
    print(f"grad_x max_abs_diff={(x.grad - x_ref.grad).abs().max().item():.6e}")
    print(f"grad_x allclose={torch.allclose(x.grad, x_ref.grad, atol=atol, rtol=rtol)}")
    print(f"grad_weight max_abs_diff={(weight.grad - weight_ref.grad).abs().max().item():.6e}")
    print(f"grad_weight allclose={torch.allclose(weight.grad, weight_ref.grad, atol=atol, rtol=rtol)}")


if __name__ == "__main__":
    main()
