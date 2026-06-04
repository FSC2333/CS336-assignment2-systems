from __future__ import annotations

import argparse
import csv
import gc
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.pytorch_attention import (
    BATCH_SIZE,
    D_MODEL_VALUES,
    DTYPES,
    SEQUENCE_LENGTHS,
    MemoryStats,
    TimingStats,
    clear_grads,
    cleanup_cuda,
    estimate_memory,
    format_mib,
    is_cuda_oom,
    make_qkv,
    summarize_memory,
    summarize_timings,
    synchronize,
)


BACKENDS = ("eager", "compiled")


class AttentionModule(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)


@dataclass(frozen=True)
class BackendResult:
    name: str
    status: str
    forward: TimingStats | None
    backward: TimingStats | None
    pre_backward_memory: MemoryStats | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    error_message: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    d_model: int
    sequence_length: int
    eager: BackendResult
    compiled: BackendResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eager and torch.compile PyTorch scaled dot-product attention.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size. Use the default 8 for the assignment report.")
    parser.add_argument("--d-models", type=int, nargs="+", default=list(D_MODEL_VALUES), help="Head embedding dimensions to sweep.")
    parser.add_argument("--sequence-lengths", "--seq-lens", type=int, nargs="+", default=list(SEQUENCE_LENGTHS), help="Sequence lengths to sweep.")
    parser.add_argument("--warmup-steps", "-w", type=int, default=10, help="Warmup iterations. For compiled attention, these trigger compilation.")
    parser.add_argument("--measurement-steps", "-n", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path for CSV results.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be at least 1 so torch.compile overhead is excluded from measurement.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")
    if any(d_model <= 0 for d_model in args.d_models):
        raise ValueError("--d-models must contain only positive integers.")
    if any(sequence_length <= 0 for sequence_length in args.sequence_lengths):
        raise ValueError("--sequence-lengths must contain only positive integers.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU because it uses torch.cuda memory and synchronization APIs.")


def build_attention(backend: str, args: argparse.Namespace, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    module = AttentionModule().to(device)
    if backend == "eager":
        return module
    if backend != "compiled":
        raise ValueError(f"Unsupported backend: {backend}")

    compile_kwargs: dict[str, Any] = {
        "backend": args.compile_backend,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
    }
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(module, **compile_kwargs)


def run_forward_warmup(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        clear_grads(q, k, v)
        output = attention(q, k, v)
        synchronize(device)
        del output


def benchmark_forward(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
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
        output = attention(q, k, v)
        synchronize(device)
        timings.append(timeit.default_timer() - start)
        del output
    return summarize_timings(timings)


def run_backward_warmup(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
) -> None:
    for _ in range(warmup_steps):
        clear_grads(q, k, v)
        output = attention(q, k, v)
        loss = output.sum()
        synchronize(device)
        loss.backward()
        synchronize(device)
        del loss, output


def benchmark_backward(
    attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
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
        output = attention(q, k, v)
        loss = output.sum()
        synchronize(device)
        pre_backward_memory.append(torch.cuda.memory_allocated(device))

        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        timings.append(timeit.default_timer() - start)
        del loss, output
    return summarize_timings(timings), summarize_memory(pre_backward_memory)


def reset_compile_state() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()


def benchmark_backend(
    *,
    backend: str,
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BackendResult:
    torch.cuda.reset_peak_memory_stats(device)
    attention = build_attention(backend, args, device)
    q, k, v = make_qkv(
        batch_size=args.batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        device=device,
        dtype=dtype,
    )

    run_forward_warmup(attention, q=q, k=k, v=v, device=device, warmup_steps=args.warmup_steps)
    forward = benchmark_forward(attention, q=q, k=k, v=v, device=device, measurement_steps=args.measurement_steps)

    run_backward_warmup(attention, q=q, k=k, v=v, device=device, warmup_steps=args.warmup_steps)
    backward, pre_backward_memory = benchmark_backward(attention, q=q, k=k, v=v, device=device, measurement_steps=args.measurement_steps)

    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    clear_grads(q, k, v)
    del q, k, v, attention
    gc.collect()
    cleanup_cuda(device)
    if backend == "compiled":
        reset_compile_state()
    return BackendResult(
        name=backend,
        status="ok",
        forward=forward,
        backward=backward,
        pre_backward_memory=pre_backward_memory,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
    )


def failed_backend_result(backend: str, status: str, message: str) -> BackendResult:
    return BackendResult(
        name=backend,
        status=status,
        forward=None,
        backward=None,
        pre_backward_memory=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        error_message=message,
    )


def run_backend(
    *,
    backend: str,
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BackendResult:
    try:
        return benchmark_backend(backend=backend, args=args, d_model=d_model, sequence_length=sequence_length, device=device, dtype=dtype)
    except RuntimeError as exc:
        cleanup_cuda(device)
        if backend == "compiled":
            reset_compile_state()
        if is_cuda_oom(exc):
            return failed_backend_result(backend, "oom", str(exc).splitlines()[0])
        raise


def seconds_or_none(result: BackendResult, mode: str) -> float | None:
    stats = result.forward if mode == "forward" else result.backward
    return stats.mean_s if stats is not None else None


def format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "OOM"
    return f"{seconds * 1_000:.3f}"


def format_speedup(eager_seconds: float | None, compiled_seconds: float | None) -> str:
    if eager_seconds is None or compiled_seconds is None:
        return "NA"
    return f"{eager_seconds / compiled_seconds:.3f}x"


def print_header(args: argparse.Namespace, device: torch.device) -> None:
    print("PyTorch attention torch.compile benchmark")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size} dtype={args.dtype}")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print(f"compile_backend={args.compile_backend} compile_mode={args.compile_mode} fullgraph={args.fullgraph} dynamic={args.dynamic}")
    print()
    columns = (
        ("backend", 10),
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


def print_backend_result(
    *,
    args: argparse.Namespace,
    result: ComparisonResult,
    backend: BackendResult,
    dtype: torch.dtype,
) -> None:
    estimate = estimate_memory(
        batch_size=args.batch_size,
        sequence_length=result.sequence_length,
        d_model=result.d_model,
        dtype=dtype,
    )
    columns = (
        (backend.name, 10),
        (str(result.d_model), 8),
        (str(result.sequence_length), 8),
        (backend.status, 8),
        (format_ms(seconds_or_none(backend, "forward")), 12),
        (format_ms(seconds_or_none(backend, "backward")), 12),
        (format_mib(backend.pre_backward_memory.max_bytes if backend.pre_backward_memory is not None else None), 13),
        (format_mib(backend.peak_allocated_bytes), 11),
        (format_mib(estimate.saved_for_backward_bytes), 14),
        (format_mib(estimate.forward_peak_bytes), 13),
    )
    print(" ".join(f"{value:>{width}}" for value, width in columns), flush=True)


def print_result(args: argparse.Namespace, result: ComparisonResult, dtype: torch.dtype) -> None:
    print_backend_result(args=args, result=result, backend=result.eager, dtype=dtype)
    print_backend_result(args=args, result=result, backend=result.compiled, dtype=dtype)


def backend_to_row(args: argparse.Namespace, comparison: ComparisonResult, backend: BackendResult, dtype: torch.dtype) -> dict[str, Any]:
    estimate = estimate_memory(
        batch_size=args.batch_size,
        sequence_length=comparison.sequence_length,
        d_model=comparison.d_model,
        dtype=dtype,
    )
    return {
        "backend": backend.name,
        "status": backend.status,
        "batch_size": args.batch_size,
        "sequence_length": comparison.sequence_length,
        "d_model": comparison.d_model,
        "dtype": args.dtype,
        "forward_mean_s": backend.forward.mean_s if backend.forward is not None else "",
        "forward_std_s": backend.forward.std_s if backend.forward is not None else "",
        "forward_min_s": backend.forward.min_s if backend.forward is not None else "",
        "forward_max_s": backend.forward.max_s if backend.forward is not None else "",
        "backward_mean_s": backend.backward.mean_s if backend.backward is not None else "",
        "backward_std_s": backend.backward.std_s if backend.backward is not None else "",
        "backward_min_s": backend.backward.min_s if backend.backward is not None else "",
        "backward_max_s": backend.backward.max_s if backend.backward is not None else "",
        "pre_backward_memory_mean_bytes": backend.pre_backward_memory.mean_bytes if backend.pre_backward_memory is not None else "",
        "pre_backward_memory_max_bytes": backend.pre_backward_memory.max_bytes if backend.pre_backward_memory is not None else "",
        "pre_backward_memory_min_bytes": backend.pre_backward_memory.min_bytes if backend.pre_backward_memory is not None else "",
        "peak_allocated_bytes": backend.peak_allocated_bytes if backend.peak_allocated_bytes is not None else "",
        "peak_reserved_bytes": backend.peak_reserved_bytes if backend.peak_reserved_bytes is not None else "",
        "estimated_attention_matrix_bytes": estimate.attention_matrix_bytes,
        "estimated_saved_for_backward_bytes": estimate.saved_for_backward_bytes,
        "estimated_live_before_backward_bytes": estimate.live_before_backward_bytes,
        "estimated_forward_peak_bytes": estimate.forward_peak_bytes,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
        "error_message": backend.error_message or "",
    }


def result_to_rows(args: argparse.Namespace, result: ComparisonResult, dtype: torch.dtype) -> list[dict[str, Any]]:
    return [
        backend_to_row(args, result, result.eager, dtype),
        backend_to_row(args, result, result.compiled, dtype),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
            eager = run_backend(backend="eager", args=args, d_model=d_model, sequence_length=sequence_length, device=device, dtype=dtype)
            compiled = run_backend(backend="compiled", args=args, d_model=d_model, sequence_length=sequence_length, device=device, dtype=dtype)
            result = ComparisonResult(d_model=d_model, sequence_length=sequence_length, eager=eager, compiled=compiled)
            results.append(result)
            print_result(args, result, dtype)

    if args.csv is not None:
        write_csv(args.csv, [row for result in results for row in result_to_rows(args, result, dtype)])
        print()
        print(f"Wrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
