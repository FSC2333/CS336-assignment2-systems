from __future__ import annotations

import argparse
import csv
import gc
import os
import socket
import timeit
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import DistributedDataParallel
from cs336_systems.sharded_optimizer import ShardedOptimizer


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class MemoryRecord:
    mode: str
    checkpoint: str
    rank: int
    allocated_bytes: int
    peak_allocated_bytes: int
    reserved_bytes: int
    peak_reserved_bytes: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    other_allocated_bytes: int
    other_peak_allocated_bytes: int


@dataclass(frozen=True)
class StepTiming:
    mode: str
    rank: int
    step: int
    total_s: float
    gradient_sync_s: float
    optimizer_step_s: float


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}
OPTIMIZER_MODES = ("full", "sharded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile optimizer-state-sharding memory accounting for CS336 Transformer LM training.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--modes", choices=(*OPTIMIZER_MODES, "both"), nargs="+", default=["both"])
    parser.add_argument("--world-size", type=int, default=2, help="Number of single-node DDP ranks/GPUs.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES mask, e.g. 0,3 for physical GPUs 0 and 3.")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-foreach", choices=("auto", "true", "false"), default="false")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--process-group-timeout-s", type=int, default=300)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--timing-csv", type=Path, default=None)
    return parser.parse_args()


def selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if "both" in args.modes:
        return OPTIMIZER_MODES
    return tuple(dict.fromkeys(args.modes))


def validate_args(args: argparse.Namespace) -> None:
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive.")
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive.")
    if args.global_batch_size % args.world_size != 0:
        raise ValueError("--global-batch-size must be divisible by --world-size.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")
    if args.process_group_timeout_s <= 0:
        raise ValueError("--process-group-timeout-s must be positive.")


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    return "nccl" if torch.cuda.is_available() else "gloo"


def find_free_port(addr: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((addr, 0))
        return int(sock.getsockname()[1])


def setup_process_group(args: argparse.Namespace, *, backend: str, rank: int, world_size: int, master_port: int) -> torch.device:
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA.")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(f"NCCL run requested {world_size} ranks, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            torch.set_num_threads(1)
            device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{args.master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.process_group_timeout_s),
    )
    return device


def distributed_barrier(device: torch.device) -> None:
    if dist.get_backend() == "nccl" and device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def clear_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def build_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> DistributedDataParallel:
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
    return DistributedDataParallel(model)


def optimizer_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.adamw_foreach != "auto":
        kwargs["foreach"] = args.adamw_foreach == "true"
    return kwargs


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module, mode: str) -> torch.optim.Optimizer:
    kwargs = optimizer_kwargs(args)
    if mode == "sharded":
        return ShardedOptimizer(model.parameters(), torch.optim.AdamW, **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def make_local_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    local_batch_size = args.global_batch_size // args.world_size
    tokens = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    return tokens, targets


def language_modeling_loss(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> torch.Tensor:
    with autocast_context(device, amp_dtype):
        logits = model(tokens)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def tensor_storage_bytes(tensors, *, device: torch.device | None = None) -> int:
    total = 0
    seen_storages: set[tuple[str, int, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        if device is not None and tensor.device != device:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key in seen_storages:
            continue
        seen_storages.add(key)
        total += storage.nbytes()
    return total


def nested_tensor_storage_bytes(value: Any, *, device: torch.device, seen_storages: set[tuple[str, int, int]]) -> int:
    if torch.is_tensor(value):
        if value.device != device:
            return 0
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes())
        if key in seen_storages:
            return 0
        seen_storages.add(key)
        return storage.nbytes()
    if isinstance(value, Mapping):
        return sum(nested_tensor_storage_bytes(item, device=device, seen_storages=seen_storages) for item in value.values())
    if isinstance(value, (tuple, list, set)):
        return sum(nested_tensor_storage_bytes(item, device=device, seen_storages=seen_storages) for item in value)
    return 0


def optimizer_state_bytes(optimizer: torch.optim.Optimizer | None, *, device: torch.device) -> int:
    if optimizer is None:
        return 0
    seen_storages: set[tuple[str, int, int]] = set()
    return nested_tensor_storage_bytes(optimizer.state, device=device, seen_storages=seen_storages)


def model_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    wrapped_module = getattr(model, "module", model)
    return list(wrapped_module.parameters())


def record_memory(
    *,
    mode: str,
    checkpoint: str,
    rank: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> MemoryRecord:
    synchronize(device)
    parameters = model_parameters(model)
    parameter_bytes = tensor_storage_bytes(parameters, device=device)
    gradient_bytes = tensor_storage_bytes((parameter.grad for parameter in parameters), device=device)
    opt_state_bytes = optimizer_state_bytes(optimizer, device=device)

    if device.type == "cuda":
        allocated_bytes = torch.cuda.memory_allocated(device)
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    else:
        allocated_bytes = peak_allocated_bytes = reserved_bytes = peak_reserved_bytes = 0

    accounted_bytes = parameter_bytes + gradient_bytes + opt_state_bytes
    return MemoryRecord(
        mode=mode,
        checkpoint=checkpoint,
        rank=rank,
        allocated_bytes=allocated_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
        reserved_bytes=reserved_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=opt_state_bytes,
        other_allocated_bytes=max(0, allocated_bytes - accounted_bytes),
        other_peak_allocated_bytes=max(0, peak_allocated_bytes - accounted_bytes),
    )


def timed_training_iteration(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss = language_modeling_loss(model, tokens, targets, device, amp_dtype)
    loss.backward()

    synchronize(device)
    gradient_sync_start = timeit.default_timer()
    model.finish_gradient_synchronization()
    gradient_sync_s = timeit.default_timer() - gradient_sync_start

    synchronize(device)
    optimizer_step_start = timeit.default_timer()
    optimizer.step()
    synchronize(device)
    optimizer_step_s = timeit.default_timer() - optimizer_step_start

    return gradient_sync_s, optimizer_step_s


def benchmark_training(
    *,
    mode: str,
    rank: int,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[StepTiming]:
    for _ in range(args.warmup_steps):
        distributed_barrier(device)
        timed_training_iteration(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
            device=device,
            amp_dtype=args.amp_dtype,
        )

    distributed_barrier(device)
    synchronize(device)

    timings = []
    for step in range(args.measurement_steps):
        distributed_barrier(device)
        synchronize(device)

        total_start = timeit.default_timer()
        gradient_sync_s, optimizer_step_s = timed_training_iteration(
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
            device=device,
            amp_dtype=args.amp_dtype,
        )
        total_s = timeit.default_timer() - total_start
        timings.append(
            StepTiming(
                mode=mode,
                rank=rank,
                step=step,
                total_s=total_s,
                gradient_sync_s=gradient_sync_s,
                optimizer_step_s=optimizer_step_s,
            )
        )

    distributed_barrier(device)
    return timings


def run_one_mode(
    rank: int,
    args: argparse.Namespace,
    config: ModelConfig,
    mode: str,
    device: torch.device,
) -> tuple[int, list[MemoryRecord], list[StepTiming]]:
    clear_device_memory(device)
    reset_peak_memory(device)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)

    model = build_model(args, config, device)
    distributed_barrier(device)
    model_init_record = record_memory(mode=mode, checkpoint="after_model_init", rank=rank, model=model, optimizer=None, device=device)

    reset_peak_memory(device)
    optimizer = build_optimizer(args, model, mode)
    tokens, targets = make_local_batch(args, device)

    optimizer.zero_grad(set_to_none=True)
    loss = language_modeling_loss(model, tokens, targets, device, args.amp_dtype)
    loss.backward()
    model.finish_gradient_synchronization()
    distributed_barrier(device)
    before_step_record = record_memory(mode=mode, checkpoint="before_optimizer_step", rank=rank, model=model, optimizer=optimizer, device=device)

    reset_peak_memory(device)
    optimizer.step()
    distributed_barrier(device)
    after_step_record = record_memory(mode=mode, checkpoint="after_optimizer_step", rank=rank, model=model, optimizer=optimizer, device=device)

    timings = benchmark_training(
        mode=mode,
        rank=rank,
        model=model,
        optimizer=optimizer,
        tokens=tokens,
        targets=targets,
        device=device,
        args=args,
    )

    num_params = sum(parameter.numel() for parameter in model_parameters(model))
    del loss
    del tokens
    del targets
    del optimizer
    del model
    clear_device_memory(device)
    return num_params, [model_init_record, before_step_record, after_step_record], timings


def worker(rank: int, args: argparse.Namespace, backend: str, modes: tuple[str, ...], master_port: int, result_queue: mp.SimpleQueue) -> None:
    device = setup_process_group(args, backend=backend, rank=rank, world_size=args.world_size, master_port=master_port)
    try:
        config = MODEL_CONFIGS[args.model_size]
        all_records = []
        all_timings = []
        num_params = 0
        for mode_idx, mode in enumerate(modes):
            if mode_idx > 0:
                distributed_barrier(device)
            num_params, local_records, local_timings = run_one_mode(rank, args, config, mode, device)
            gathered_records: list[list[MemoryRecord] | None] = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered_records, local_records)
            gathered_timings: list[list[StepTiming] | None] = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered_timings, local_timings)
            if rank == 0:
                for records in gathered_records:
                    if records is not None:
                        all_records.extend(records)
                for timings in gathered_timings:
                    if timings is not None:
                        all_timings.extend(timings)

        if rank == 0:
            result_queue.put((num_params, all_records, all_timings))
    finally:
        dist.destroy_process_group()


def run_benchmark(args: argparse.Namespace, backend: str, modes: tuple[str, ...]) -> tuple[int, list[MemoryRecord], list[StepTiming]]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    master_port = find_free_port(args.master_addr)
    mp.spawn(worker, args=(args, backend, modes, master_port, result_queue), nprocs=args.world_size, join=True)
    return result_queue.get()


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def max_record_value(records: list[MemoryRecord], checkpoint: str, mode: str, attr: str) -> int:
    return max(getattr(record, attr) for record in records if record.mode == mode and record.checkpoint == checkpoint)


def mean_record_value(records: list[MemoryRecord], checkpoint: str, mode: str, attr: str) -> float:
    values = [getattr(record, attr) for record in records if record.mode == mode and record.checkpoint == checkpoint]
    return sum(values) / len(values)


def timing_values(timings: list[StepTiming], mode: str, step: int, attr: str) -> list[float]:
    return [getattr(timing, attr) for timing in timings if timing.mode == mode and timing.step == step]


def average_rank_max_timing(timings: list[StepTiming], mode: str, attr: str, measurement_steps: int) -> float:
    rank_max_values = []
    for step in range(measurement_steps):
        values = timing_values(timings, mode, step, attr)
        rank_max_values.append(max(values))
    return sum(rank_max_values) / len(rank_max_values)


def print_results(
    args: argparse.Namespace,
    backend: str,
    modes: tuple[str, ...],
    num_params: int,
    records: list[MemoryRecord],
    timings: list[StepTiming],
) -> None:
    config = MODEL_CONFIGS[args.model_size]
    local_batch_size = args.global_batch_size // args.world_size
    device_label = "cuda:0-" + str(args.world_size - 1) if backend == "nccl" else backend
    if args.cuda_visible_devices is not None:
        device_label += f" (CUDA_VISIBLE_DEVICES={args.cuda_visible_devices})"

    print("Optimizer state sharding memory accounting setting")
    print(f"model_size={args.model_size} config={config} parameters={num_params:,}")
    print(f"modes={','.join(modes)} world_size={args.world_size} device={device_label} backend={backend}")
    print(f"global_batch_size={args.global_batch_size} local_batch_size={local_batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(
        f"optimizer=torch-adamw adamw_foreach={args.adamw_foreach} amp_dtype={args.amp_dtype} "
        f"warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}"
    )
    print()

    print("Rank-max peak memory by checkpoint")
    print(f"{'mode':<8} {'checkpoint':<23} {'peak alloc GiB':>14} {'mean peak GiB':>13} {'current alloc GiB':>17} {'reserved peak GiB':>17}")
    print("-" * 96)
    checkpoints = ("after_model_init", "before_optimizer_step", "after_optimizer_step")
    for mode in modes:
        for checkpoint in checkpoints:
            print(
                f"{mode:<8} "
                f"{checkpoint:<23} "
                f"{bytes_to_gib(max_record_value(records, checkpoint, mode, 'peak_allocated_bytes')):14.3f} "
                f"{bytes_to_gib(int(mean_record_value(records, checkpoint, mode, 'peak_allocated_bytes'))):13.3f} "
                f"{bytes_to_gib(max_record_value(records, checkpoint, mode, 'allocated_bytes')):17.3f} "
                f"{bytes_to_gib(max_record_value(records, checkpoint, mode, 'peak_reserved_bytes')):17.3f}"
            )

    print()
    print("Per-rank current allocation breakdown at each checkpoint")
    print(
        f"{'mode':<8} {'checkpoint':<23} {'rank':>4} {'current GiB':>12} {'params GiB':>11} "
        f"{'grads GiB':>10} {'opt state GiB':>14} {'other GiB':>11} {'peak GiB':>10}"
    )
    print("-" * 111)
    for record in sorted(records, key=lambda item: (modes.index(item.mode), checkpoints.index(item.checkpoint), item.rank)):
        print(
            f"{record.mode:<8} "
            f"{record.checkpoint:<23} "
            f"{record.rank:4d} "
            f"{bytes_to_gib(record.allocated_bytes):12.3f} "
            f"{bytes_to_gib(record.parameter_bytes):11.3f} "
            f"{bytes_to_gib(record.gradient_bytes):10.3f} "
            f"{bytes_to_gib(record.optimizer_state_bytes):14.3f} "
            f"{bytes_to_gib(record.other_allocated_bytes):11.3f} "
            f"{bytes_to_gib(record.peak_allocated_bytes):10.3f}"
        )

    print()
    print("Training iteration timing after optimizer state initialization")
    print(
        f"{'mode':<8} {'iter':>6} {'total max ms':>14} {'grad sync max ms':>16} "
        f"{'opt step max ms':>16} {'opt step %':>10} {'total mean ms':>14} {'opt mean ms':>12}"
    )
    print("-" * 104)
    for mode in modes:
        for step in range(args.measurement_steps):
            total_s = timing_values(timings, mode, step, "total_s")
            gradient_sync_s = timing_values(timings, mode, step, "gradient_sync_s")
            optimizer_step_s = timing_values(timings, mode, step, "optimizer_step_s")
            max_total_s = max(total_s)
            max_optimizer_step_s = max(optimizer_step_s)
            print(
                f"{mode:<8} "
                f"{step:6d} "
                f"{max_total_s * 1_000:14.3f} "
                f"{max(gradient_sync_s) * 1_000:16.3f} "
                f"{max_optimizer_step_s * 1_000:16.3f} "
                f"{(max_optimizer_step_s / max_total_s) * 100:10.2f} "
                f"{(sum(total_s) / len(total_s)) * 1_000:14.3f} "
                f"{(sum(optimizer_step_s) / len(optimizer_step_s)) * 1_000:12.3f}"
            )

    print()
    print(f"{'mode':<8} {'avg total max ms':>18} {'avg grad sync ms':>17} {'avg opt step ms':>16} {'total vs baseline':>18}")
    print("-" * 82)
    baseline_mode = "full" if "full" in modes else modes[0]
    baseline_total_s = average_rank_max_timing(timings, baseline_mode, "total_s", args.measurement_steps)
    for mode in modes:
        avg_total_s = average_rank_max_timing(timings, mode, "total_s", args.measurement_steps)
        avg_gradient_sync_s = average_rank_max_timing(timings, mode, "gradient_sync_s", args.measurement_steps)
        avg_optimizer_step_s = average_rank_max_timing(timings, mode, "optimizer_step_s", args.measurement_steps)
        print(
            f"{mode:<8} "
            f"{avg_total_s * 1_000:18.3f} "
            f"{avg_gradient_sync_s * 1_000:17.3f} "
            f"{avg_optimizer_step_s * 1_000:16.3f} "
            f"{baseline_total_s / avg_total_s:18.3f}x"
        )


def write_csv(path: Path, records: list[MemoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_timing_csv(path: Path, timings: list[StepTiming]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(timings[0]).keys()))
        writer.writeheader()
        for timing in timings:
            writer.writerow(asdict(timing))


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    validate_args(args)
    modes = selected_modes(args)
    backend = resolve_backend(args)
    num_params, records, timings = run_benchmark(args, backend, modes)
    print_results(args, backend, modes, num_params, records, timings)
    if args.csv is not None:
        write_csv(args.csv, records)
        print(f"\nWrote CSV results to {args.csv}")
    if args.timing_csv is not None:
        write_timing_csv(args.timing_csv, timings)
        print(f"Wrote timing CSV results to {args.timing_csv}")


if __name__ == "__main__":
    main()
