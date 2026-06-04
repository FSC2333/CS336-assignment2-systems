from __future__ import annotations

import argparse
import csv
import gc
import statistics
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW as BasicsAdamW


BENCHMARK_MODES = ("forward", "training_step")
BACKENDS = ("eager", "compiled")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class TimingStats:
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    num_steps: int


@dataclass(frozen=True)
class BenchmarkResult:
    model_size: str
    backend: str
    mode: str
    status: str
    timing: TimingStats | None
    num_params: int | None
    error_message: str | None = None


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eager and torch.compile end-to-end Transformer benchmarks.")
    parser.add_argument("--model-size", choices=(*MODEL_CONFIGS.keys(), "all"), default="small")
    parser.add_argument("--mode", choices=(*BENCHMARK_MODES, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model. Only valid with a single --model-size.")
    parser.add_argument("--d-ff", type=int, default=None, help="Override d_ff. Only valid with a single --model-size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override layer count. Only valid with a single --model-size.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override head count. Only valid with a single --model-size.")
    parser.add_argument("--warmup-steps", "-w", type=int, default=5, help="Warmup iterations. For compiled models, these trigger compilation.")
    parser.add_argument("--measurement-steps", "-n", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("basics-adamw", "torch-adamw"), default="basics-adamw")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write results as CSV.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be at least 1 so torch.compile overhead is excluded from measurement.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")
    overrides = (args.d_model, args.d_ff, args.num_layers, args.num_heads)
    if args.model_size == "all" and any(value is not None for value in overrides):
        raise ValueError("Hyperparameter overrides are only valid with a single --model-size.")


def selected_model_sizes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.model_size == "all":
        return tuple(MODEL_CONFIGS)
    return (args.model_size,)


def selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.mode == "all":
        return BENCHMARK_MODES
    return (args.mode,)


def resolve_model_config(args: argparse.Namespace, model_size: str) -> ModelConfig:
    values = asdict(MODEL_CONFIGS[model_size])
    overrides = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return ModelConfig(**values)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_compile_state() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def build_base_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    model = model.to(device=device, dtype=DTYPES[args.dtype])
    model.train()
    return model


def compile_model(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    compile_kwargs: dict[str, Any] = {
        "backend": args.compile_backend,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
    }
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(model, **compile_kwargs)


def build_optimizer(args: argparse.Namespace, parameters) -> torch.optim.Optimizer:
    if args.optimizer == "basics-adamw":
        return BasicsAdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    return tokens, targets


def language_modeling_loss(model: torch.nn.Module, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def make_step_fn(
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> Callable[[], None]:
    if mode == "forward":

        def forward_step() -> None:
            model(tokens)

        return forward_step

    if mode == "training_step":

        def training_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = language_modeling_loss(model, tokens, targets)
            loss.backward()
            optimizer.step()

        return training_step

    raise ValueError(f"Unsupported benchmark mode: {mode}")


def benchmark_step(
    step_fn: Callable[[], None],
    *,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
) -> TimingStats:
    for _ in range(warmup_steps):
        step_fn()
        synchronize(device)

    timings = []
    for _ in range(measurement_steps):
        synchronize(device)
        start = timeit.default_timer()
        step_fn()
        synchronize(device)
        timings.append(timeit.default_timer() - start)

    return TimingStats(
        mean_s=statistics.fmean(timings),
        std_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_s=min(timings),
        max_s=max(timings),
        num_steps=len(timings),
    )


def benchmark_backend_mode(
    *,
    args: argparse.Namespace,
    model_size: str,
    config: ModelConfig,
    backend: str,
    mode: str,
    device: torch.device,
) -> BenchmarkResult:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    base_model = build_base_model(args, config, device)
    num_params = sum(p.numel() for p in base_model.parameters())
    optimizer = build_optimizer(args, base_model.parameters())
    model = compile_model(base_model, args) if backend == "compiled" else base_model
    tokens, targets = make_batch(args, device)

    step_fn = make_step_fn(mode, model, optimizer, tokens, targets)
    timing = benchmark_step(
        step_fn,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
    )

    del step_fn, tokens, targets, optimizer, model, base_model
    cleanup(device)
    if backend == "compiled":
        reset_compile_state()

    return BenchmarkResult(
        model_size=model_size,
        backend=backend,
        mode=mode,
        status="ok",
        timing=timing,
        num_params=num_params,
    )


def failed_result(model_size: str, backend: str, mode: str, status: str, message: str) -> BenchmarkResult:
    return BenchmarkResult(
        model_size=model_size,
        backend=backend,
        mode=mode,
        status=status,
        timing=None,
        num_params=None,
        error_message=message,
    )


def run_one(
    *,
    args: argparse.Namespace,
    model_size: str,
    config: ModelConfig,
    backend: str,
    mode: str,
    device: torch.device,
) -> BenchmarkResult:
    try:
        return benchmark_backend_mode(
            args=args,
            model_size=model_size,
            config=config,
            backend=backend,
            mode=mode,
            device=device,
        )
    except RuntimeError as exc:
        cleanup(device)
        if backend == "compiled":
            reset_compile_state()
        if is_cuda_oom(exc):
            return failed_result(model_size, backend, mode, "oom", str(exc).splitlines()[0])
        raise


def format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "OOM"
    return f"{seconds * 1_000:.3f}"


def timing_mean(result: BenchmarkResult) -> float | None:
    return result.timing.mean_s if result.timing is not None else None


def format_speedup(eager: BenchmarkResult | None, compiled: BenchmarkResult | None) -> str:
    if eager is None or compiled is None:
        return "NA"
    eager_s = timing_mean(eager)
    compiled_s = timing_mean(compiled)
    if eager_s is None or compiled_s is None:
        return "NA"
    return f"{eager_s / compiled_s:.3f}x"


def print_header(args: argparse.Namespace, device: torch.device) -> None:
    print("End-to-end Transformer torch.compile benchmark")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"batch_size={args.batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(f"dtype={args.dtype} optimizer={args.optimizer}")
    print(f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print(f"compile_backend={args.compile_backend} compile_mode={args.compile_mode} fullgraph={args.fullgraph} dynamic={args.dynamic}")
    print()
    columns = (
        ("model", 8),
        ("mode", 14),
        ("backend", 10),
        ("status", 8),
        ("mean_ms", 12),
        ("std_ms", 12),
        ("min_ms", 12),
        ("max_ms", 12),
        ("steps", 7),
        ("speedup", 9),
    )
    print(" ".join(f"{name:>{width}}" for name, width in columns))


def print_result(result: BenchmarkResult, speedup: str = "") -> None:
    timing = result.timing
    columns = (
        (result.model_size, 8),
        (result.mode, 14),
        (result.backend, 10),
        (result.status, 8),
        (format_ms(timing.mean_s if timing is not None else None), 12),
        (format_ms(timing.std_s if timing is not None else None), 12),
        (format_ms(timing.min_s if timing is not None else None), 12),
        (format_ms(timing.max_s if timing is not None else None), 12),
        (str(timing.num_steps) if timing is not None else "0", 7),
        (speedup, 9),
    )
    print(" ".join(f"{value:>{width}}" for value, width in columns), flush=True)


def result_to_row(args: argparse.Namespace, config: ModelConfig, result: BenchmarkResult, speedup: str) -> dict[str, Any]:
    timing = result.timing
    return {
        "model_size": result.model_size,
        "mode": result.mode,
        "backend": result.backend,
        "status": result.status,
        "mean_s": timing.mean_s if timing is not None else "",
        "std_s": timing.std_s if timing is not None else "",
        "min_s": timing.min_s if timing is not None else "",
        "max_s": timing.max_s if timing is not None else "",
        "num_steps": timing.num_steps if timing is not None else "",
        "speedup_eager_over_compiled": speedup,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "device": args.device,
        "dtype": args.dtype,
        "optimizer": args.optimizer,
        "num_params": result.num_params if result.num_params is not None else "",
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "compile_backend": args.compile_backend,
        "compile_mode": args.compile_mode,
        "fullgraph": args.fullgraph,
        "dynamic": args.dynamic,
        "error_message": result.error_message or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)

    rows = []
    print_header(args, device)
    for model_size in selected_model_sizes(args):
        config = resolve_model_config(args, model_size)
        for mode in selected_modes(args):
            eager = run_one(args=args, model_size=model_size, config=config, backend="eager", mode=mode, device=device)
            compiled = run_one(args=args, model_size=model_size, config=config, backend="compiled", mode=mode, device=device)
            speedup = format_speedup(eager, compiled)
            print_result(eager)
            print_result(compiled, speedup=speedup)
            rows.append(result_to_row(args, config, eager, ""))
            rows.append(result_to_row(args, config, compiled, speedup))

    if args.csv is not None:
        write_csv(args.csv, rows)
        print()
        print(f"Wrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
