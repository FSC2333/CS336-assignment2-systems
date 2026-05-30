from __future__ import annotations

import argparse
import csv
import statistics
import timeit
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW as BasicsAdamW


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class TimingResult:
    mode: str
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    num_steps: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

BENCHMARK_MODES = ("forward", "forward_backward", "training_step")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end benchmark for the CS336 basics Transformer language model.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="small", help="Named model size from the assignment handout.")
    parser.add_argument("--mode", choices=(*BENCHMARK_MODES, "all"), default="all", help="Which training fragment to benchmark.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None, help="Override the named model d_model.")
    parser.add_argument("--d-ff", type=int, default=None, help="Override the named model d_ff.")
    parser.add_argument("--num-layers", type=int, default=None, help="Override the named model layer count.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override the named model head count.")
    parser.add_argument("--warmup-steps", "-w", type=int, default=5)
    parser.add_argument("--measurement-steps", "-n", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("basics-adamw", "torch-adamw"), default="basics-adamw")
    parser.add_argument("--torch-compile", action="store_true", help="Compile the model with torch.compile before benchmarking.")
    parser.add_argument("--nvtx", action="store_true", help="Wrap measured steps in NVTX ranges for Nsight Systems.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write the timing table as CSV.")
    return parser.parse_args()


def resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    values = asdict(MODEL_CONFIGS[args.model_size])
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


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def build_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> torch.nn.Module:
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
    if args.torch_compile:
        model = torch.compile(model)
    return model


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "basics-adamw":
        return BasicsAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]: # 随机生成输入tokens和目标tokens
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    return tokens, targets
    # tokens:  [batch_size, context_length]
    # targets: [batch_size, context_length]

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

    if mode == "forward_backward":

        def forward_backward_step() -> None:
            model.zero_grad(set_to_none=True)
            loss = language_modeling_loss(model, tokens, targets)
            loss.backward()

        return forward_backward_step

    if mode == "training_step":

        def training_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            loss = language_modeling_loss(model, tokens, targets)
            loss.backward()
            optimizer.step()

        return training_step

    raise ValueError(f"Unsupported benchmark mode: {mode}")


def benchmark_step(
    mode: str,
    step_fn: Callable[[], None],
    *,
    warmup_steps: int,
    measurement_steps: int,
    device: torch.device,
    use_nvtx: bool,
) -> TimingResult:
    for _ in range(warmup_steps):
        step_fn()
        synchronize(device)

    timings = []
    for step_idx in range(measurement_steps):
        synchronize(device)
        start = timeit.default_timer()
        with nvtx_range(f"{mode}/step_{step_idx}", enabled=use_nvtx):
            step_fn()
            synchronize(device)
        timings.append(timeit.default_timer() - start)

    return TimingResult(
        mode=mode,
        mean_s=statistics.fmean(timings),
        std_s=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        min_s=min(timings),
        max_s=max(timings),
        num_steps=len(timings),
    )


def selected_modes(mode: str) -> tuple[str, ...]:
    if mode == "all":
        return BENCHMARK_MODES
    return (mode,)


def format_ms(seconds: float) -> str:
    return f"{seconds * 1_000:10.3f}"


def print_results(args: argparse.Namespace, config: ModelConfig, results: list[TimingResult], num_params: int) -> None:
    print(f"model_size={args.model_size} config={config}")
    print(f"batch_size={args.batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(f"device={args.device} dtype={args.dtype} optimizer={args.optimizer} parameters={num_params:,}")
    print()
    print(f"{'mode':<18} {'mean_ms':>10} {'std_ms':>10} {'min_ms':>10} {'max_ms':>10} {'steps':>7}")
    for result in results:
        print(f"{result.mode:<18} {format_ms(result.mean_s)} {format_ms(result.std_s)} {format_ms(result.min_s)} {format_ms(result.max_s)} {result.num_steps:7d}")

    by_mode = {result.mode: result for result in results}
    if all(mode in by_mode for mode in BENCHMARK_MODES):
        backward_estimate = by_mode["forward_backward"].mean_s - by_mode["forward"].mean_s
        optimizer_estimate = by_mode["training_step"].mean_s - by_mode["forward_backward"].mean_s
        print()
        print("derived component estimates from mean timings:")
        print(f"backward ~= forward_backward - forward = {backward_estimate * 1_000:.3f} ms")
        print(f"optimizer ~= training_step - forward_backward = {optimizer_estimate * 1_000:.3f} ms")


def write_csv(path: Path, args: argparse.Namespace, config: ModelConfig, results: list[TimingResult], num_params: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_size",
                "mode",
                "mean_s",
                "std_s",
                "min_s",
                "max_s",
                "num_steps",
                "batch_size",
                "context_length",
                "vocab_size",
                "d_model",
                "d_ff",
                "num_layers",
                "num_heads",
                "device",
                "dtype",
                "optimizer",
                "num_params",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model_size": args.model_size,
                    "mode": result.mode,
                    "mean_s": result.mean_s,
                    "std_s": result.std_s,
                    "min_s": result.min_s,
                    "max_s": result.max_s,
                    "num_steps": result.num_steps,
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
                    "num_params": num_params,
                }
            )


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.nvtx and device.type != "cuda":
        raise ValueError("--nvtx requires a CUDA device")

    config = resolve_model_config(args)
    model = build_model(args, config, device)
    optimizer = build_optimizer(args, model)
    tokens, targets = make_batch(args, device)
    num_params = sum(p.numel() for p in model.parameters())

    results = []
    for mode in selected_modes(args.mode):
        step_fn = make_step_fn(mode, model, optimizer, tokens, targets)
        results.append(
            benchmark_step(
                mode,
                step_fn,
                warmup_steps=args.warmup_steps,
                measurement_steps=args.measurement_steps,
                device=device,
                use_nvtx=args.nvtx,
            )
        )

    print_results(args, config, results, num_params)
    if args.csv is not None:
        write_csv(args.csv, args, config, results, num_params)


if __name__ == "__main__":
    main()
