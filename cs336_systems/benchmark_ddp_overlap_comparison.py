from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.benchmark_minimal_ddp_flat import (
    MODEL_CONFIGS,
    benchmark_training,
    build_optimizer,
    clear_device_memory,
    distributed_barrier,
    find_free_port,
    make_local_batch,
    resolve_backend,
    setup_process_group,
    summarize_timings,
    validate_args,
)
from cs336_systems.ddp import DistributedDataParallel, FlatDistributedDataParallel
from cs336_systems.ddp_overlap import OverlapDistributedDataParallel


DDP_MODES = ("naive", "flat", "overlap")
DDP_CLASSES = {
    "naive": DistributedDataParallel,
    "flat": FlatDistributedDataParallel,
    "overlap": OverlapDistributedDataParallel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare naive, flat-gradient, and overlapped per-parameter DDP communication.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--modes", choices=(*DDP_MODES, "all", "both"), nargs="+", default=["all"])
    parser.add_argument("--world-size", type=int, default=2, help="Number of single-node DDP ranks/GPUs.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES mask, e.g. 0,3 for physical GPUs 0 and 3.")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--optimizer", choices=("torch-adamw", "basics-adamw", "sgd"), default="torch-adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--process-group-timeout-s", type=int, default=300)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def selected_modes(args: argparse.Namespace) -> tuple[str, ...]:
    if "all" in args.modes or "both" in args.modes:
        return DDP_MODES
    return tuple(dict.fromkeys(args.modes))


def build_model(args: argparse.Namespace, device: torch.device, mode: str) -> torch.nn.Module:
    config = MODEL_CONFIGS[args.model_size]
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
    return DDP_CLASSES[mode](model)


def worker(rank: int, args: argparse.Namespace, backend: str, modes: tuple[str, ...], master_port: int, result_queue: mp.SimpleQueue) -> None:
    device = setup_process_group(args, backend=backend, rank=rank, world_size=args.world_size, master_port=master_port)
    try:
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + rank)
        tokens, targets = make_local_batch(args, device)

        all_summaries = []
        num_params = 0
        for mode_idx, mode in enumerate(modes):
            torch.manual_seed(args.seed + rank)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + rank)

            model = build_model(args, device, mode)
            optimizer = build_optimizer(args, model)
            num_params = sum(parameter.numel() for parameter in model.module.parameters())

            local_timings = benchmark_training(
                mode=mode,
                rank=rank,
                model=model,
                optimizer=optimizer,
                tokens=tokens,
                targets=targets,
                device=device,
                args=args,
            )

            gathered_timings = [None for _ in range(args.world_size)]
            dist.all_gather_object(gathered_timings, local_timings)
            if rank == 0:
                rank_timings = [timings for timings in gathered_timings if timings is not None]
                all_summaries.extend(summarize_timings(mode, rank_timings, args.measurement_steps))

            del model
            del optimizer
            clear_device_memory(device)
            if mode_idx < len(modes) - 1:
                distributed_barrier(device)

        if rank == 0:
            result_queue.put((num_params, all_summaries))
    finally:
        dist.destroy_process_group()


def run_benchmark(args: argparse.Namespace, backend: str, modes: tuple[str, ...]):
    ctx = mp.get_context("spawn")
    result_queue = ctx.SimpleQueue()
    master_port = find_free_port(args.master_addr)
    mp.spawn(worker, args=(args, backend, modes, master_port, result_queue), nprocs=args.world_size, join=True)
    return result_queue.get()


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def print_results(args: argparse.Namespace, backend: str, modes: tuple[str, ...], num_params: int, summaries) -> None:
    config = MODEL_CONFIGS[args.model_size]
    local_batch_size = args.global_batch_size // args.world_size
    device_label = "cuda:0-" + str(args.world_size - 1) if backend == "nccl" else backend
    if args.cuda_visible_devices is not None:
        device_label += f" (CUDA_VISIBLE_DEVICES={args.cuda_visible_devices})"

    print("DDP communication overlap benchmark setting")
    print(f"model_size={args.model_size} config={config} parameters={num_params:,}")
    print(f"modes={','.join(modes)} world_size={args.world_size} device={device_label} backend={backend}")
    print(f"global_batch_size={args.global_batch_size} local_batch_size={local_batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(f"optimizer={args.optimizer} amp_dtype={args.amp_dtype} warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print("For overlap, sync wait is the remaining wait in finish_gradient_synchronization(); most communication is launched during backward hooks.")
    print()
    print(f"{'mode':<8} {'iter':>6} {'total max ms':>14} {'sync wait max ms':>16} {'sync wait %':>11} {'total mean ms':>14} {'sync wait mean ms':>18}")
    print("-" * 105)
    for summary in summaries:
        print(
            f"{summary.mode:<8} "
            f"{summary.step:6d} "
            f"{summary.max_total_ms:14.3f} "
            f"{summary.max_communication_ms:16.3f} "
            f"{summary.max_communication_fraction * 100:11.2f} "
            f"{summary.mean_total_ms:14.3f} "
            f"{summary.mean_communication_ms:18.3f}"
        )

    by_mode = {mode: [summary for summary in summaries if summary.mode == mode] for mode in modes}
    naive_avg_total_ms = None
    if "naive" in by_mode and by_mode["naive"]:
        naive_avg_total_ms = mean([row.max_total_ms for row in by_mode["naive"]])

    print()
    print(f"{'mode':<8} {'avg total max ms':>18} {'avg sync wait ms':>18} {'avg sync wait %':>16} {'total vs naive':>16}")
    print("-" * 82)
    for mode in modes:
        rows = by_mode[mode]
        avg_total_ms = mean([row.max_total_ms for row in rows])
        avg_sync_wait_ms = mean([row.max_communication_ms for row in rows])
        avg_sync_wait_fraction = mean([row.max_communication_fraction for row in rows]) * 100
        if naive_avg_total_ms is None:
            speedup = "n/a"
        else:
            speedup = f"{naive_avg_total_ms / avg_total_ms:.3f}x"
        print(f"{mode:<8} {avg_total_ms:18.3f} {avg_sync_wait_ms:18.3f} {avg_sync_wait_fraction:16.2f} {speedup:>16}")


def write_csv(path: Path, summaries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    validate_args(args)
    modes = selected_modes(args)
    backend = resolve_backend(args)
    num_params, summaries = run_benchmark(args, backend, modes)
    print_results(args, backend, modes, num_params, summaries)
    if args.csv is not None:
        write_csv(args.csv, summaries)
        print(f"\nWrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
