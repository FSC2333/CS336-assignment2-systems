from __future__ import annotations

import argparse
import csv
import os
import timeit
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.benchmark_optimizer_state_sharding_accounting import (
    MODEL_CONFIGS,
    MemoryRecord,
    ModelConfig,
    StepTiming,
    autocast_context,
    bytes_to_gib,
    clear_device_memory,
    distributed_barrier,
    find_free_port,
    make_local_batch,
    optimizer_kwargs,
    record_memory,
    reset_peak_memory,
    setup_process_group,
    synchronize,
)
from cs336_systems.ddp import DistributedDataParallel
from cs336_systems.fsdp import FullyShardedDataParallel


MODES = ("ddp", "fsdp")
COMPUTE_DTYPES = {
    "none": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DDP vs FSDP memory accounting for CS336 Transformer LM training.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--modes", choices=(*MODES, "both"), nargs="+", default=["both"])
    parser.add_argument("--world-size", type=int, default=2, help="Number of single-node DDP/FSDP ranks.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES mask, e.g. 0,1.")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--compute-dtype", choices=tuple(COMPUTE_DTYPES), default="none", help="FSDP gathered-weight dtype.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adamw-foreach", choices=("auto", "true", "false"), default="false")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--process-group-timeout-s", type=int, default=300)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--timing-csv", type=Path, default=None)
    return parser.parse_args()


def selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if "both" in args.modes:
        return MODES
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


def build_base_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> BasicsTransformerLM:
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
    return model


def wrap_model(args: argparse.Namespace, base_model: BasicsTransformerLM, mode: str) -> torch.nn.Module:
    if mode == "ddp":
        return DistributedDataParallel(base_model)
    if mode == "fsdp":
        return FullyShardedDataParallel(base_model, compute_dtype=COMPUTE_DTYPES[args.compute_dtype])
    raise ValueError(f"Unknown mode: {mode}")


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), **optimizer_kwargs(args))


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


def timed_training_iteration(
    *,
    model: torch.nn.Module,
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
    model: torch.nn.Module,
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

    base_model = build_base_model(args, config, device)
    full_num_params = sum(parameter.numel() for parameter in base_model.parameters())
    model = wrap_model(args, base_model, mode)
    distributed_barrier(device)
    model_init_record = record_memory(mode=mode, checkpoint="after_model_init", rank=rank, model=model, optimizer=None, device=device)

    reset_peak_memory(device)
    optimizer = build_optimizer(args, model)
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

    del loss
    del tokens
    del targets
    del optimizer
    del model
    clear_device_memory(device)
    return full_num_params, [model_init_record, before_step_record, after_step_record], timings


def worker(rank: int, args: argparse.Namespace, backend: str, modes: tuple[str, ...], master_port: int, result_queue: mp.SimpleQueue) -> None:
    device = setup_process_group(args, backend=backend, rank=rank, world_size=args.world_size, master_port=master_port)
    try:
        config = MODEL_CONFIGS[args.model_size]
        all_records: list[MemoryRecord] = []
        all_timings: list[StepTiming] = []
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


def timing_values(timings: list[StepTiming], mode: str, step: int, attr: str) -> list[float]:
    return [getattr(timing, attr) for timing in timings if timing.mode == mode and timing.step == step]


def average_rank_max_timing(timings: list[StepTiming], mode: str, attr: str, measurement_steps: int) -> float:
    rank_max_values = []
    for step in range(measurement_steps):
        values = timing_values(timings, mode, step, attr)
        if values:
            rank_max_values.append(max(values))
    return sum(rank_max_values) / len(rank_max_values)


def max_record_value(records: list[MemoryRecord], checkpoint: str, mode: str, attr: str) -> int:
    return max(getattr(record, attr) for record in records if record.mode == mode and record.checkpoint == checkpoint)


def rank_max_persistent_state_bytes(records: list[MemoryRecord], checkpoint: str, mode: str) -> int:
    values = []
    for record in records:
        if record.mode == mode and record.checkpoint == checkpoint:
            values.append(record.parameter_bytes + record.gradient_bytes + record.optimizer_state_bytes)
    return max(values)


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

    print("FSDP memory accounting setting")
    print(f"model_size={args.model_size} config={config} parameters={num_params:,}")
    print(f"modes={','.join(modes)} world_size={args.world_size} device={device_label} backend={backend}")
    print(f"global_batch_size={args.global_batch_size} local_batch_size={local_batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(
        f"optimizer=torch-adamw adamw_foreach={args.adamw_foreach} amp_dtype={args.amp_dtype} "
        f"fsdp_compute_dtype={args.compute_dtype} warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}"
    )
    print()

    checkpoints = ("after_model_init", "before_optimizer_step", "after_optimizer_step")
    print("Rank-max peak memory by checkpoint")
    print(f"{'mode':<8} {'checkpoint':<23} {'peak alloc GiB':>14} {'current alloc GiB':>17} {'persistent state GiB':>20}")
    print("-" * 88)
    for mode in modes:
        for checkpoint in checkpoints:
            persistent_state_bytes = rank_max_persistent_state_bytes(records, checkpoint, mode)
            print(
                f"{mode:<8} "
                f"{checkpoint:<23} "
                f"{bytes_to_gib(max_record_value(records, checkpoint, mode, 'peak_allocated_bytes')):14.3f} "
                f"{bytes_to_gib(max_record_value(records, checkpoint, mode, 'allocated_bytes')):17.3f} "
                f"{bytes_to_gib(persistent_state_bytes):20.3f}"
            )

    if "ddp" in modes and "fsdp" in modes:
        checkpoint = "after_optimizer_step"
        ddp_state = rank_max_persistent_state_bytes(records, checkpoint, "ddp")
        fsdp_state = rank_max_persistent_state_bytes(records, checkpoint, "fsdp")
        saving = 1.0 - (fsdp_state / ddp_state)
        ideal_saving = 1.0 - (1.0 / args.world_size)
        print()
        print("Persistent model-state savings, ignoring transient all-gather buffers")
        print(f"measured_rank_max_saving={saving * 100:.2f}% ideal_if_every_parameter_were_sharded={ideal_saving * 100:.2f}%")

    print()
    print("Per-rank allocation breakdown")
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
    print(f"{'mode':<8} {'avg total max ms':>18} {'avg grad sync ms':>17} {'avg opt step ms':>16} {'total vs ddp':>14}")
    print("-" * 78)
    baseline_total_s = average_rank_max_timing(timings, "ddp" if "ddp" in modes else modes[0], "total_s", args.measurement_steps)
    for mode in modes:
        avg_total_s = average_rank_max_timing(timings, mode, "total_s", args.measurement_steps)
        avg_gradient_sync_s = average_rank_max_timing(timings, mode, "gradient_sync_s", args.measurement_steps)
        avg_optimizer_step_s = average_rank_max_timing(timings, mode, "optimizer_step_s", args.measurement_steps)
        print(
            f"{mode:<8} "
            f"{avg_total_s * 1_000:18.3f} "
            f"{avg_gradient_sync_s * 1_000:17.3f} "
            f"{avg_optimizer_step_s * 1_000:16.3f} "
            f"{baseline_total_s / avg_total_s:14.3f}x"
        )


def write_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


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
        print(f"\nWrote memory records to {args.csv}")
    if args.timing_csv is not None:
        write_csv(args.timing_csv, timings)
        print(f"Wrote timing records to {args.timing_csv}")


if __name__ == "__main__":
    main()
