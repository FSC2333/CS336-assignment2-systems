from __future__ import annotations

import argparse
import csv
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


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
class PrecisionResult:
    model_size: str
    precision: str
    forward: TimingStats
    forward_backward: TimingStats
    backward_estimate_s: float
    num_params: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

ASSIGNMENT_MODEL_SIZES = ("small", "medium", "large", "xl", "10B")
PRECISIONS = ("fp32", "bf16_mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs BF16 autocast mixed precision for CS336 Transformer models.")
    parser.add_argument("--model-size", choices=(*MODEL_CONFIGS.keys(), "all"), default="small")
    parser.add_argument("--precision", choices=(*PRECISIONS, "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model. Only valid with a single --model-size.")
    parser.add_argument("--d-ff", type=int, default=None, help="Override d_ff. Only valid with a single --model-size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override layer count. Only valid with a single --model-size.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override head count. Only valid with a single --model-size.")
    parser.add_argument("--warmup-steps", "-w", type=int, default=5)
    parser.add_argument("--measurement-steps", "-n", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def selected_model_sizes(args: argparse.Namespace) -> tuple[str, ...]:
    if args.model_size == "all":
        return ASSIGNMENT_MODEL_SIZES
    return (args.model_size,)


def selected_precisions(args: argparse.Namespace) -> tuple[str, ...]:
    if args.precision == "both":
        return PRECISIONS
    return (args.precision,)


def resolve_model_config(args: argparse.Namespace, model_size: str) -> ModelConfig:
    overrides = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    if args.model_size == "all" and any(value is not None for value in overrides.values()):
        raise ValueError("Hyperparameter overrides are only valid with a single --model-size.")

    values = asdict(MODEL_CONFIGS[model_size])
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return ModelConfig(**values)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision_context(device: torch.device, precision: str):
    if precision == "bf16_mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def build_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    if args.torch_compile:
        model = torch.compile(model)
    return model


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    return tokens, targets


def forward_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    with precision_context(device, precision):
        logits = model(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def forward_step(model: torch.nn.Module, tokens: torch.Tensor, device: torch.device, precision: str) -> None:
    with precision_context(device, precision):
        model(tokens)


def forward_backward_step(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
) -> None:
    model.zero_grad(set_to_none=True)
    loss = forward_loss(model, tokens, targets, device, precision)
    loss.backward()


def benchmark_callable(
    step_fn,
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


def benchmark_precision(
    *,
    args: argparse.Namespace,
    model_size: str,
    config: ModelConfig,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
    num_params: int,
) -> PrecisionResult:
    forward = benchmark_callable(
        lambda: forward_step(model, tokens, device, precision),
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
    )
    forward_backward = benchmark_callable(
        lambda: forward_backward_step(model, tokens, targets, device, precision),
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=device,
    )
    return PrecisionResult(
        model_size=model_size,
        precision=precision,
        forward=forward,
        forward_backward=forward_backward,
        backward_estimate_s=forward_backward.mean_s - forward.mean_s,
        num_params=num_params,
    )


def format_ms(seconds: float) -> str:
    return f"{seconds * 1_000:.3f}"


def print_model_results(args: argparse.Namespace, model_size: str, config: ModelConfig, results: list[PrecisionResult]) -> None:
    print(f"model_size={model_size} config={config}")
    print(f"batch_size={args.batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(f"device={args.device} parameters={results[0].num_params:,}")
    print()
    columns = (
        ("precision", 14),
        ("forward_ms", 12),
        ("forward_std", 12),
        ("fwd_bwd_ms", 12),
        ("fwd_bwd_std", 12),
        ("backward_est_ms", 16),
        ("steps", 7),
    )
    print(" ".join(f"{name:>{width}}" for name, width in columns))
    for result in results:
        row = (
            result.precision,
            format_ms(result.forward.mean_s),
            format_ms(result.forward.std_s),
            format_ms(result.forward_backward.mean_s),
            format_ms(result.forward_backward.std_s),
            format_ms(result.backward_estimate_s),
            str(result.forward.num_steps),
        )
        print(" ".join(f"{value:>{width}}" for value, (_, width) in zip(row, columns, strict=True)))

    by_precision = {result.precision: result for result in results}
    if "fp32" in by_precision and "bf16_mixed" in by_precision:
        fp32 = by_precision["fp32"]
        mixed = by_precision["bf16_mixed"]
        print()
        print("bf16_mixed speedup vs fp32:")
        print(f"forward: {fp32.forward.mean_s / mixed.forward.mean_s:.3f}x")
        print(f"forward_backward: {fp32.forward_backward.mean_s / mixed.forward_backward.mean_s:.3f}x")
        print(f"backward_estimate: {fp32.backward_estimate_s / mixed.backward_estimate_s:.3f}x")
    print()


def result_to_rows(args: argparse.Namespace, config: ModelConfig, result: PrecisionResult) -> list[dict[str, Any]]:
    common = {
        "model_size": result.model_size,
        "precision": result.precision,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "device": args.device,
        "num_params": result.num_params,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
    }
    return [
        {
            **common,
            "mode": "forward",
            "mean_s": result.forward.mean_s,
            "std_s": result.forward.std_s,
            "min_s": result.forward.min_s,
            "max_s": result.forward.max_s,
        },
        {
            **common,
            "mode": "forward_backward",
            "mean_s": result.forward_backward.mean_s,
            "std_s": result.forward_backward.std_s,
            "min_s": result.forward_backward.min_s,
            "max_s": result.forward_backward.max_s,
        },
        {
            **common,
            "mode": "backward_estimate",
            "mean_s": result.backward_estimate_s,
            "std_s": "",
            "min_s": "",
            "max_s": "",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive")

    device = torch.device(args.device)
    all_rows = []
    for model_size in selected_model_sizes(args):
        torch.manual_seed(args.seed)
        config = resolve_model_config(args, model_size)
        model = build_model(args, config, device)
        tokens, targets = make_batch(args, device)
        num_params = sum(p.numel() for p in model.parameters())

        results = []
        for precision in selected_precisions(args):
            results.append(
                benchmark_precision(
                    args=args,
                    model_size=model_size,
                    config=config,
                    model=model,
                    tokens=tokens,
                    targets=targets,
                    device=device,
                    precision=precision,
                    num_params=num_params,
                )
            )

        print_model_results(args, model_size, config, results)
        for result in results:
            all_rows.extend(result_to_rows(args, config, result))

        del model, tokens, targets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.csv is not None:
        write_csv(args.csv, all_rows)


if __name__ == "__main__":
    main()
