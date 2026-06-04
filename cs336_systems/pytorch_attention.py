from __future__ import annotations

import argparse
import csv
import gc
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import scaled_dot_product_attention


BATCH_SIZE = 8
D_MODEL_VALUES = (16, 32, 64, 128)
SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class TimingStats:
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    num_steps: int


@dataclass(frozen=True)
class MemoryStats:
    mean_bytes: float
    max_bytes: int
    min_bytes: int


@dataclass(frozen=True)
class MemoryEstimate:
    qkv_bytes: int
    output_bytes: int
    attention_matrix_bytes: int
    saved_for_backward_bytes: int
    live_before_backward_bytes: int
    forward_peak_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    d_model: int
    sequence_length: int
    status: str
    forward: TimingStats | None
    backward: TimingStats | None
    pre_backward_memory: MemoryStats | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    estimate: MemoryEstimate
    oom_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vanilla PyTorch scaled dot-product attention for CS336 Assignment 2.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size. Use the default 8 for the assignment report.")
    parser.add_argument("--d-models", type=int, nargs="+", default=list(D_MODEL_VALUES), help="Head embedding dimensions to sweep.")
    parser.add_argument("--sequence-lengths", "--seq-lens", type=int, nargs="+", default=list(SEQUENCE_LENGTHS), help="Sequence lengths to sweep.")
    parser.add_argument("--warmup-steps", "-w", type=int, default=10)
    parser.add_argument("--measurement-steps", "-n", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None, help="Optional path for CSV results.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")
    if any(d_model <= 0 for d_model in args.d_models):
        raise ValueError("--d-models must contain only positive integers.")
    if any(sequence_length <= 0 for sequence_length in args.sequence_lengths):
        raise ValueError("--sequence-lengths must contain only positive integers.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU because it uses torch.cuda memory and synchronization APIs.")


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def bytes_to_mib(num_bytes: float) -> float:
    return num_bytes / 1024**2


def format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "OOM"
    return f"{seconds * 1_000:.3f}"


def format_mib(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "OOM"
    return f"{bytes_to_mib(num_bytes):.2f}"


def summarize_timings(timings: list[float]) -> TimingStats:
    return TimingStats(
        mean_s=statistics.fmean(timings),
        std_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_s=min(timings),
        max_s=max(timings),
        num_steps=len(timings),
    )


def summarize_memory(values: list[int]) -> MemoryStats:
    return MemoryStats(
        mean_bytes=statistics.fmean(values),
        max_bytes=max(values),
        min_bytes=min(values),
    )


def clear_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    q.grad = None
    k.grad = None
    v.grad = None


def make_qkv(
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, d_model)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)


def run_forward_warmup(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        clear_grads(q, k, v)
        output = attention_forward(q, k, v)
        synchronize(device)
        del output


def benchmark_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    measurement_steps: int,
) -> TimingStats:
    timings = []
    for _ in range(measurement_steps):
        clear_grads(q, k, v)
        synchronize(device)
        start = timeit.default_timer()
        output = attention_forward(q, k, v)
        synchronize(device)
        timings.append(timeit.default_timer() - start)
        del output
    return summarize_timings(timings)


def run_backward_warmup(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        clear_grads(q, k, v)
        output = attention_forward(q, k, v)
        loss = output.sum()
        synchronize(device)
        loss.backward()
        synchronize(device)
        del loss, output


def benchmark_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    measurement_steps: int,
) -> tuple[TimingStats, MemoryStats]:
    timings = []
    pre_backward_memory = []
    for _ in range(measurement_steps):
        clear_grads(q, k, v)
        output = attention_forward(q, k, v)
        loss = output.sum()
        synchronize(device)
        pre_backward_memory.append(torch.cuda.memory_allocated(device))

        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        timings.append(timeit.default_timer() - start)
        del loss, output
    return summarize_timings(timings), summarize_memory(pre_backward_memory)


def estimate_memory(
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    dtype: torch.dtype,
) -> MemoryEstimate:
    element_size = torch.empty((), dtype=dtype).element_size()
    qkv_bytes = 3 * batch_size * sequence_length * d_model * element_size
    output_bytes = batch_size * sequence_length * d_model * element_size
    attention_matrix_bytes = batch_size * sequence_length * sequence_length * element_size
    saved_for_backward_bytes = qkv_bytes + attention_matrix_bytes
    live_before_backward_bytes = saved_for_backward_bytes + output_bytes
    forward_peak_bytes = qkv_bytes + output_bytes + 2 * attention_matrix_bytes
    return MemoryEstimate(
        qkv_bytes=qkv_bytes,
        output_bytes=output_bytes,
        attention_matrix_bytes=attention_matrix_bytes,
        saved_for_backward_bytes=saved_for_backward_bytes,
        live_before_backward_bytes=live_before_backward_bytes,
        forward_peak_bytes=forward_peak_bytes,
    )


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def benchmark_config(
    *,
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BenchmarkResult:
    estimate = estimate_memory(
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        dtype=dtype,
    )
    torch.cuda.reset_peak_memory_stats(device)
    q, k, v = make_qkv(
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        device=device,
        dtype=dtype,
    )

    run_forward_warmup(q=q, k=k, v=v, device=device, warmup_steps=args.warmup_steps)
    forward = benchmark_forward(q=q, k=k, v=v, device=device, measurement_steps=args.measurement_steps)

    run_backward_warmup(q=q, k=k, v=v, device=device, warmup_steps=args.warmup_steps)
    backward, pre_backward_memory = benchmark_backward(q=q, k=k, v=v, device=device, measurement_steps=args.measurement_steps)

    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    clear_grads(q, k, v)
    del q, k, v
    cleanup_cuda(device)
    return BenchmarkResult(
        d_model=d_model,
        sequence_length=sequence_length,
        status="ok",
        forward=forward,
        backward=backward,
        pre_backward_memory=pre_backward_memory,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        estimate=estimate,
    )


def oom_result(
    *,
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
    dtype: torch.dtype,
    oom_message: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        d_model=d_model,
        sequence_length=sequence_length,
        status="oom",
        forward=None,
        backward=None,
        pre_backward_memory=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        estimate=estimate_memory(
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            d_model=d_model,
            dtype=dtype,
        ),
        oom_message=oom_message,
    )


def result_to_row(args: argparse.Namespace, result: BenchmarkResult) -> dict[str, Any]:
    forward = result.forward
    backward = result.backward
    pre_mem = result.pre_backward_memory
    return {
        "status": result.status,
        "batch_size": args.batch_size,
        "sequence_length": result.sequence_length,
        "d_model": result.d_model,
        "dtype": args.dtype,
        "forward_mean_s": forward.mean_s if forward is not None else "",
        "forward_std_s": forward.std_s if forward is not None else "",
        "forward_min_s": forward.min_s if forward is not None else "",
        "forward_max_s": forward.max_s if forward is not None else "",
        "backward_mean_s": backward.mean_s if backward is not None else "",
        "backward_std_s": backward.std_s if backward is not None else "",
        "backward_min_s": backward.min_s if backward is not None else "",
        "backward_max_s": backward.max_s if backward is not None else "",
        "pre_backward_memory_mean_bytes": pre_mem.mean_bytes if pre_mem is not None else "",
        "pre_backward_memory_max_bytes": pre_mem.max_bytes if pre_mem is not None else "",
        "pre_backward_memory_min_bytes": pre_mem.min_bytes if pre_mem is not None else "",
        "peak_allocated_bytes": result.peak_allocated_bytes if result.peak_allocated_bytes is not None else "",
        "peak_reserved_bytes": result.peak_reserved_bytes if result.peak_reserved_bytes is not None else "",
        "estimated_attention_matrix_bytes": result.estimate.attention_matrix_bytes,
        "estimated_saved_for_backward_bytes": result.estimate.saved_for_backward_bytes,
        "estimated_live_before_backward_bytes": result.estimate.live_before_backward_bytes,
        "estimated_forward_peak_bytes": result.estimate.forward_peak_bytes,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "oom_message": result.oom_message or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_header(args: argparse.Namespace, device: torch.device) -> None:
    print("PyTorch attention benchmark")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size} dtype={args.dtype}")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print()
    columns = (
        ("d_model", 8),
        ("seq_len", 8),
        ("status", 8),
        ("fwd_ms", 12),
        ("bwd_ms", 12),
        ("pre_bwd_mib", 13),
        ("peak_mib", 11),
        ("est_saved_mib", 14),
        ("est_peak_mib", 13),
    )
    print(" ".join(f"{name:>{width}}" for name, width in columns))


def print_result(result: BenchmarkResult) -> None:
    columns = (
        (str(result.d_model), 8),
        (str(result.sequence_length), 8),
        (result.status, 8),
        (format_ms(result.forward.mean_s if result.forward is not None else None), 12),
        (format_ms(result.backward.mean_s if result.backward is not None else None), 12),
        (format_mib(result.pre_backward_memory.max_bytes if result.pre_backward_memory is not None else None), 13),
        (format_mib(result.peak_allocated_bytes), 11),
        (format_mib(result.estimate.saved_for_backward_bytes), 14),
        (format_mib(result.estimate.forward_peak_bytes), 13),
    )
    print(" ".join(f"{value:>{width}}" for value, width in columns), flush=True)


def print_memory_analysis(results: list[BenchmarkResult]) -> None:
    oom_results = [result for result in results if result.status == "oom"]
    if not oom_results:
        print()
        print("No OOM configurations were encountered.")
        return

    smallest = min(oom_results, key=lambda result: result.estimate.forward_peak_bytes)
    estimate = smallest.estimate
    print()
    print("Smallest OOM configuration by estimated vanilla-attention forward peak:")
    print(f"d_model={smallest.d_model} sequence_length={smallest.sequence_length}")
    print(f"attention matrix B*S*S={format_mib(estimate.attention_matrix_bytes)} MiB")
    print(f"saved for backward ~= Q,K,V + softmax(QK^T)={format_mib(estimate.saved_for_backward_bytes)} MiB")
    print(f"live before backward ~= saved + output={format_mib(estimate.live_before_backward_bytes)} MiB")
    print(f"forward peak estimate ~= Q,K,V,output + scores + softmax(scores)={format_mib(estimate.forward_peak_bytes)} MiB")
    print("The quadratic B*S*S term dominates, so saved attention memory grows as O(sequence_length^2).")
    print("FlashAttention removes this cost by recomputing attention tiles in backward instead of materializing and saving the full attention matrix.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    validate_args(args, device)
    dtype = DTYPES[args.dtype]

    results = []
    print_header(args, device)
    for d_model in args.d_models:
        for sequence_length in args.sequence_lengths:
            try:
                result = benchmark_config(
                    args=args,
                    d_model=d_model,
                    sequence_length=sequence_length,
                    device=device,
                    dtype=dtype,
                )
            except RuntimeError as exc:
                if not is_cuda_oom(exc):
                    raise
                cleanup_cuda(device)
                result = oom_result(
                    args=args,
                    d_model=d_model,
                    sequence_length=sequence_length,
                    dtype=dtype,
                    oom_message=str(exc).splitlines()[0],
                )
            results.append(result)
            print_result(result)

    if args.csv is not None:
        write_csv(args.csv, [result_to_row(args, result) for result in results])
        print()
        print(f"Wrote CSV results to {args.csv}")

    print_memory_analysis(results)


if __name__ == "__main__":
    main()
