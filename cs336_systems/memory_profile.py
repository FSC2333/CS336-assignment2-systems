from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

AMP_DTYPES = {
    "none": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA memory profiler for CS336 Assignment 2 memory_profiling.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=("inference", "train_step"), required=True)
    parser.add_argument("--profile-memory", action="store_true", help="Record and dump a PyTorch CUDA memory snapshot.")
    parser.add_argument("--memory-snapshot-dir", type=Path, default=Path("memory_snapshots"))
    parser.add_argument("--memory-max-entries", type=int, default=1_000_000)
    parser.add_argument("--amp-dtype", choices=sorted(AMP_DTYPES), default="none")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("basics-adamw", "torch-adamw"), default="basics-adamw")
    return parser.parse_args()


def resolve_model_config(model_size: str) -> ModelConfig:
    return MODEL_CONFIGS[model_size]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp_context(device: torch.device, amp_dtype: str):
    dtype = AMP_DTYPES[amp_dtype]
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    return model.to(device=device, dtype=torch.float32)


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "basics-adamw":
        return BasicsAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    return input_ids, targets


def inference_step(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device, amp_dtype: str) -> None:
    model.eval()
    with torch.inference_mode():
        with amp_context(device, amp_dtype):
            model(input_ids)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, amp_dtype):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_warmup(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> None:
    for _ in range(args.warmup_steps):
        if args.mode == "inference":
            inference_step(model, input_ids, device, args.amp_dtype)
        else:
            assert optimizer is not None
            train_step(model, optimizer, input_ids, targets, device, args.amp_dtype)
        synchronize(device)


def run_profiled_step(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float | None:
    if args.mode == "inference":
        inference_step(model, input_ids, device, args.amp_dtype)
        synchronize(device)
        return None

    assert optimizer is not None
    loss = train_step(model, optimizer, input_ids, targets, device, args.amp_dtype)
    synchronize(device)
    return loss


def memory_snapshot_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    precision = "fp32" if args.amp_dtype == "none" else args.amp_dtype
    stem = f"memory_snapshot_{args.model_size}_ctx{args.context_length}_{args.mode}_bs{args.batch_size}_{precision}"
    return args.memory_snapshot_dir / f"{stem}.pickle", args.memory_snapshot_dir / f"{stem}.json"


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / 1024**2


def make_summary(
    *,
    args: argparse.Namespace,
    config: ModelConfig,
    snapshot_path: Path | None,
    loss: float | None,
    num_params: int,
) -> dict[str, Any]:
    max_allocated = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    precision = "fp32" if args.amp_dtype == "none" else args.amp_dtype
    return {
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "model_size": args.model_size,
        "config": asdict(config),
        "mode": args.mode,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "precision": precision,
        "amp_dtype": args.amp_dtype,
        "num_params": num_params,
        "loss": loss,
        "max_memory_allocated_bytes": max_allocated,
        "max_memory_allocated_mib": bytes_to_mib(max_allocated),
        "max_memory_reserved_bytes": max_reserved,
        "max_memory_reserved_mib": bytes_to_mib(max_reserved),
        "warmup_steps": args.warmup_steps,
        "profile_memory": args.profile_memory,
    }


def print_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    print(f"mode={summary['mode']}")
    print(f"context_length={summary['context_length']}")
    print(f"batch_size={summary['batch_size']}")
    print(f"precision={summary['precision']}")
    if summary["loss"] is not None:
        print(f"loss={summary['loss']:.6f}")
    print(f"max_memory_allocated={summary['max_memory_allocated_bytes']:,} bytes ({summary['max_memory_allocated_mib']:.2f} MiB)")
    print(f"max_memory_reserved={summary['max_memory_reserved_bytes']:,} bytes ({summary['max_memory_reserved_mib']:.2f} MiB)")
    if summary["snapshot_path"] is not None:
        print(f"snapshot_path={summary['snapshot_path']}")
    if summary_path is not None:
        print(f"summary_path={summary_path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling, but torch.cuda.is_available() is False.")
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required for memory profiling, but --device was {args.device!r}.")
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be at least 1 so warm-up is excluded from the memory snapshot.")
    if args.memory_max_entries <= 0:
        raise ValueError("--memory-max-entries must be positive.")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    validate_args(args, device)

    snapshot_path, summary_path = memory_snapshot_paths(args)
    if args.profile_memory:
        args.memory_snapshot_dir.mkdir(parents=True, exist_ok=True)
    else:
        snapshot_path = None
        summary_path = args.memory_snapshot_dir / f"memory_summary_{args.model_size}_ctx{args.context_length}_{args.mode}_bs{args.batch_size}.json"

    try:
        config = resolve_model_config(args.model_size)
        model = build_model(args, config, device)
        optimizer = build_optimizer(args, model) if args.mode == "train_step" else None
        input_ids, targets = make_batch(args, device)
        num_params = sum(p.numel() for p in model.parameters())

        run_warmup(args=args, model=model, optimizer=optimizer, input_ids=input_ids, targets=targets, device=device)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        loss = None
        history_enabled = False
        try:
            if args.profile_memory:
                torch.cuda.memory._record_memory_history(max_entries=args.memory_max_entries)
                history_enabled = True

            loss = run_profiled_step(args=args, model=model, optimizer=optimizer, input_ids=input_ids, targets=targets, device=device)

            if args.profile_memory:
                assert snapshot_path is not None
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
        finally:
            if history_enabled:
                torch.cuda.memory._record_memory_history(enabled=None)

        summary = make_summary(args=args, config=config, snapshot_path=snapshot_path, loss=loss, num_params=num_params)
        write_json(summary_path, summary)
        print_summary(summary, summary_path)
    except torch.cuda.OutOfMemoryError as exc:
        message = (
            f"CUDA OOM while running model_size={args.model_size}, context_length={args.context_length}, "
            f"mode={args.mode}, batch_size={args.batch_size}, precision={'fp32' if args.amp_dtype == 'none' else args.amp_dtype}. "
            "Try reducing --batch-size."
        )
        raise RuntimeError(message) from exc


if __name__ == "__main__":
    main()
