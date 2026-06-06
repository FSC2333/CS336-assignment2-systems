from __future__ import annotations

import argparse
import csv
import gc
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton

from cs336_systems.flash_attention import FlashAttentionTritonAutogradFunction


BATCH_SIZE = 1
SEQUENCE_LENGTHS = tuple(2**i for i in range(7, 17))
D_MODEL_VALUES = (16, 32, 64, 128)
DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
BACKENDS = ("triton", "pytorch")


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    dtype_name: str
    sequence_length: int
    d_model: int
    q_tile_size: int | None
    k_tile_size: int | None
    forward_ms: float | None
    backward_ms: float | None
    forward_backward_ms: float | None
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention-2 forward/backward against vanilla PyTorch causal attention.")
    parser.add_argument("--sequence-lengths", "--seq-lens", type=int, nargs="+", default=list(SEQUENCE_LENGTHS))
    parser.add_argument("--d-models", type=int, nargs="+", default=list(D_MODEL_VALUES))
    parser.add_argument("--dtypes", choices=(*sorted(DTYPES), "all"), nargs="+", default=["all"])
    parser.add_argument("--backends", choices=(*BACKENDS, "all"), nargs="+", default=["all"])
    parser.add_argument("--warmup-ms", type=int, default=25, help="Warmup duration passed to triton.testing.do_bench.")
    parser.add_argument("--rep-ms", type=int, default=100, help="Measurement duration passed to triton.testing.do_bench.")
    parser.add_argument("--q-tile-size", type=int, default=None, help="Override Triton query tile size.")
    parser.add_argument("--k-tile-size", type=int, default=None, help="Override Triton key tile size.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def expand_choices(values: list[str], all_values: tuple[str, ...]) -> tuple[str, ...]:
    if "all" in values:
        return all_values
    return tuple(values)


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    if any(sequence_length <= 0 for sequence_length in args.sequence_lengths):
        raise ValueError("--sequence-lengths must contain positive integers.")
    if any(d_model <= 0 for d_model in args.d_models):
        raise ValueError("--d-models must contain positive integers.")
    if args.warmup_ms < 0:
        raise ValueError("--warmup-ms must be non-negative.")
    if args.rep_ms <= 0:
        raise ValueError("--rep-ms must be positive.")
    if args.q_tile_size is not None and args.q_tile_size <= 0:
        raise ValueError("--q-tile-size must be positive.")
    if args.k_tile_size is not None and args.k_tile_size <= 0:
        raise ValueError("--k-tile-size must be positive.")


def clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def make_qkv(
    *,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (BATCH_SIZE, sequence_length, d_model)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.ones((1, sequence_length, sequence_length), device=device, dtype=torch.bool).tril()


def clear_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    q.grad = None
    k.grad = None
    v.grad = None


def vanilla_pytorch_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = torch.where(causal_mask, scores, torch.full((), -1.0e6, device=q.device, dtype=scores.dtype))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def triton_flash_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return FlashAttentionTritonAutogradFunction.apply(q, k, v, True)


def choose_triton_tile_sizes(d_model: int, q_tile_size: int | None, k_tile_size: int | None) -> tuple[int, int]:
    if q_tile_size is not None and k_tile_size is not None:
        return q_tile_size, k_tile_size

    if d_model >= 128:
        default_q_tile_size, default_k_tile_size = 16, 32
    elif d_model >= 64:
        default_q_tile_size, default_k_tile_size = 16, 64
    else:
        default_q_tile_size, default_k_tile_size = 32, 64

    return q_tile_size or default_q_tile_size, k_tile_size or default_k_tile_size


def set_triton_tile_sizes(q_tile_size: int, k_tile_size: int) -> None:
    FlashAttentionTritonAutogradFunction.Q_TILE_SIZE = q_tile_size
    FlashAttentionTritonAutogradFunction.K_TILE_SIZE = k_tile_size


def do_bench_ms(fn: Callable[[], Any], *, warmup_ms: int, rep_ms: int) -> float:
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, fast_flush=False))
    except TypeError as error:
        if "fast_flush" not in str(error):
            raise
        return float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms))


def benchmark_forward(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    def run() -> None:
        output = attention(q, k, v)
        del output

    return do_bench_ms(run, warmup_ms=warmup_ms, rep_ms=rep_ms)


def benchmark_backward(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    output = attention(q, k, v)
    grad_output = torch.randn_like(output)

    def run() -> None:
        clear_grads(q, k, v)
        output.backward(grad_output, retain_graph=True)

    return do_bench_ms(run, warmup_ms=warmup_ms, rep_ms=rep_ms)


def benchmark_forward_backward(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    grad_output = torch.randn_like(q)

    def run() -> None:
        clear_grads(q, k, v)
        output = attention(q, k, v)
        output.backward(grad_output)
        del output

    return do_bench_ms(run, warmup_ms=warmup_ms, rep_ms=rep_ms)


def benchmark_backend(
    *,
    backend: str,
    dtype_name: str,
    dtype: torch.dtype,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    warmup_ms: int,
    rep_ms: int,
    q_tile_size: int | None,
    k_tile_size: int | None,
) -> BenchmarkRow:
    effective_q_tile_size = None
    effective_k_tile_size = None

    try:
        q, k, v = make_qkv(sequence_length=sequence_length, d_model=d_model, dtype=dtype, device=device)
        causal_mask = None

        if backend == "triton":
            effective_q_tile_size, effective_k_tile_size = choose_triton_tile_sizes(d_model, q_tile_size, k_tile_size)
            set_triton_tile_sizes(effective_q_tile_size, effective_k_tile_size)
            attention = triton_flash_causal_attention
        elif backend == "pytorch":
            causal_mask = make_causal_mask(sequence_length, device)

            def attention(q_arg: torch.Tensor, k_arg: torch.Tensor, v_arg: torch.Tensor, causal_mask_arg: torch.Tensor = causal_mask) -> torch.Tensor:
                return vanilla_pytorch_causal_attention(q_arg, k_arg, v_arg, causal_mask_arg)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        forward_ms = benchmark_forward(attention, q, k, v, warmup_ms=warmup_ms, rep_ms=rep_ms)
        backward_ms = benchmark_backward(attention, q, k, v, warmup_ms=warmup_ms, rep_ms=rep_ms)
        forward_backward_ms = benchmark_forward_backward(attention, q, k, v, warmup_ms=warmup_ms, rep_ms=rep_ms)

        clear_grads(q, k, v)
        clear_cuda(device)

        return BenchmarkRow(
            backend=backend,
            dtype_name=dtype_name,
            sequence_length=sequence_length,
            d_model=d_model,
            q_tile_size=effective_q_tile_size,
            k_tile_size=effective_k_tile_size,
            forward_ms=forward_ms,
            backward_ms=backward_ms,
            forward_backward_ms=forward_backward_ms,
            status="ok",
        )
    except RuntimeError as error:
        clear_cuda(device)
        status = "oom" if is_cuda_oom(error) else "error"
        return BenchmarkRow(
            backend=backend,
            dtype_name=dtype_name,
            sequence_length=sequence_length,
            d_model=d_model,
            q_tile_size=effective_q_tile_size,
            k_tile_size=effective_k_tile_size,
            forward_ms=None,
            backward_ms=None,
            forward_backward_ms=None,
            status=status,
            error=str(error).splitlines()[0],
        )


def format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def format_status(row: BenchmarkRow) -> str:
    if row.error is None:
        return row.status
    message = row.error
    if len(message) > 96:
        message = f"{message[:93]}..."
    return f"{row.status}: {message}"


def row_to_dict(row: BenchmarkRow) -> dict[str, Any]:
    return {
        "backend": row.backend,
        "dtype": row.dtype_name,
        "sequence_length": row.sequence_length,
        "d_model": row.d_model,
        "batch_size": BATCH_SIZE,
        "causal": True,
        "q_tile_size": row.q_tile_size,
        "k_tile_size": row.k_tile_size,
        "forward_ms": row.forward_ms,
        "backward_ms": row.backward_ms,
        "forward_backward_ms": row.forward_backward_ms,
        "status": row.status,
        "error": row.error,
    }


def print_table(rows: list[BenchmarkRow]) -> None:
    headers = ("backend", "dtype", "seq", "d", "Bq", "Bk", "fwd ms", "bwd ms", "fwd+bwd ms", "status")
    table = [
        (
            row.backend,
            row.dtype_name,
            str(row.sequence_length),
            str(row.d_model),
            "-" if row.q_tile_size is None else str(row.q_tile_size),
            "-" if row.k_tile_size is None else str(row.k_tile_size),
            format_ms(row.forward_ms),
            format_ms(row.backward_ms),
            format_ms(row.forward_backward_ms),
            format_status(row),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for values in table:
        widths = [max(width, len(value)) for width, value in zip(widths, values, strict=True)]

    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)))
    print("-+-".join("-" * width for width in widths))
    for values in table:
        print(" | ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)))


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    dictionaries = [row_to_dict(row) for row in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dictionaries[0].keys()))
        writer.writeheader()
        writer.writerows(dictionaries)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    validate_args(args, device)

    dtype_names = expand_choices(args.dtypes, tuple(DTYPES))
    backends = expand_choices(args.backends, BACKENDS)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("FlashAttention-2 benchmark")
    print(f"device={torch.cuda.get_device_name(device)} batch_size={BATCH_SIZE} causal=True")
    print(f"sequence_lengths={list(args.sequence_lengths)} d_models={list(args.d_models)} dtypes={list(dtype_names)} backends={list(backends)}")
    print(f"do_bench warmup_ms={args.warmup_ms} rep_ms={args.rep_ms}")

    rows = []
    for dtype_name in dtype_names:
        dtype = DTYPES[dtype_name]
        for sequence_length in args.sequence_lengths:
            for d_model in args.d_models:
                for backend in backends:
                    row = benchmark_backend(
                        backend=backend,
                        dtype_name=dtype_name,
                        dtype=dtype,
                        sequence_length=sequence_length,
                        d_model=d_model,
                        device=device,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        q_tile_size=args.q_tile_size,
                        k_tile_size=args.k_tile_size,
                    )
                    rows.append(row)
                    print_table([row])

    print()
    print_table(rows)

    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"Wrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
