from __future__ import annotations

import argparse
import csv
import os
import socket
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW as BasicsAdamW
from cs336_systems.ddp import DistributedDataParallel


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class LocalStepTiming:
    rank: int
    step: int
    total_s: float
    communication_s: float


@dataclass(frozen=True)
class StepTimingSummary:
    step: int
    mean_total_ms: float
    max_total_ms: float
    mean_communication_ms: float
    max_communication_ms: float
    max_communication_fraction: float


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=128, d_ff=512, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training and gradient communication time for CS336 Transformer LMs.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--world-size", type=int, default=2, help="Number of single-node DDP ranks/GPUs.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES mask, e.g. 2,3 for physical GPUs 2 and 3.")
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--optimizer", choices=("torch-adamw", "basics-adamw"), default="torch-adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--process-group-timeout-s", type=int, default=300)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


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
            raise RuntimeError(f"NCCL benchmark requested {world_size} ranks, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
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


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "basics-adamw":
        return BasicsAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


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


def training_step(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = language_modeling_loss(model, tokens, targets, device, amp_dtype)
    loss.backward()

    synchronize(device)
    communication_start = timeit.default_timer()
    model.finish_gradient_synchronization()
    communication_s = timeit.default_timer() - communication_start

    optimizer.step()
    synchronize(device)
    return communication_s


def benchmark_training(
    *,
    rank: int,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> list[LocalStepTiming]:
    for _ in range(args.warmup_steps):
        distributed_barrier(device)
        training_step(model=model, optimizer=optimizer, tokens=tokens, targets=targets, device=device, amp_dtype=args.amp_dtype)

    distributed_barrier(device)
    synchronize(device)

    timings = []
    for step in range(args.measurement_steps):
        distributed_barrier(device)
        synchronize(device)

        total_start = timeit.default_timer()
        communication_s = training_step(model=model, optimizer=optimizer, tokens=tokens, targets=targets, device=device, amp_dtype=args.amp_dtype)
        total_s = timeit.default_timer() - total_start

        timings.append(LocalStepTiming(rank=rank, step=step, total_s=total_s, communication_s=communication_s))

    distributed_barrier(device)
    return timings


def summarize_timings(rank_timings: list[list[LocalStepTiming]], measurement_steps: int) -> list[StepTimingSummary]:
    summaries = []
    for step in range(measurement_steps):
        total_s = [timings[step].total_s for timings in rank_timings]
        communication_s = [timings[step].communication_s for timings in rank_timings]
        max_total_s = max(total_s)
        max_communication_s = max(communication_s)
        summaries.append(
            StepTimingSummary(
                step=step,
                mean_total_ms=statistics.fmean(total_s) * 1_000,
                max_total_ms=max_total_s * 1_000,
                mean_communication_ms=statistics.fmean(communication_s) * 1_000,
                max_communication_ms=max_communication_s * 1_000,
                max_communication_fraction=max_communication_s / max_total_s if max_total_s > 0 else 0.0,
            )
        )
    return summaries


def worker(rank: int, args: argparse.Namespace, backend: str, master_port: int, result_queue: mp.SimpleQueue) -> None:
    device = setup_process_group(args, backend=backend, rank=rank, world_size=args.world_size, master_port=master_port) 
    # 返回对应 rank 的设备（CPU 或 GPU）。这个函数还会初始化分布式环境，让所有 rank 之间可以通信。
    try:
        #给当前 rank 设置随机种子。
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + rank) #如果用 CUDA，也设置 CUDA 随机种子。

        config = MODEL_CONFIGS[args.model_size]
        model = build_model(args, config, device)
        # 注意 DistributedDataParallel(model) 初始化时会 broadcast rank 0 的模型参数，
        # 所以虽然不同 rank 初始 seed 不同，最后所有 rank 的模型参数会同步成 rank 0 的。
        optimizer = build_optimizer(args, model)
        tokens, targets = make_local_batch(args, device)

        local_timings = benchmark_training(
            rank=rank,
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            targets=targets,
            device=device,
            args=args,
        )

        gathered_timings: list[list[LocalStepTiming] | None] = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathered_timings, local_timings)

        if rank == 0:
            rank_timings = [timings for timings in gathered_timings if timings is not None]
            summaries = summarize_timings(rank_timings, args.measurement_steps)
            num_params = sum(parameter.numel() for parameter in model.module.parameters())
            result_queue.put((num_params, summaries))
    finally:
        dist.destroy_process_group()


def run_benchmark(args: argparse.Namespace, backend: str) -> tuple[int, list[StepTimingSummary]]:
    ctx = mp.get_context("spawn") #指定 multiprocessing 启动子进程的方式为 spawn：启动一个全新的 Python 解释器进程，然后重新 import 你的代码，再执行目标函数
    # 为什么这里要用 spawn？因为我们在跑 CUDA / distributed training。
    # CUDA 不适合用 fork 方式继承父进程状态，容易出现奇怪的问题，比如 CUDA context 被复制、NCCL 初始化异常、死锁等。
    result_queue = ctx.SimpleQueue()
    master_port = find_free_port(args.master_addr)
    mp.spawn(worker, args=(args, backend, master_port, result_queue), nprocs=args.world_size, join=True)
    return result_queue.get()


def print_results(args: argparse.Namespace, backend: str, num_params: int, summaries: list[StepTimingSummary]) -> None:
    config = MODEL_CONFIGS[args.model_size]
    local_batch_size = args.global_batch_size // args.world_size
    device_label = "cuda:0-" + str(args.world_size - 1) if backend == "nccl" else backend
    if args.cuda_visible_devices is not None:
        device_label += f" (CUDA_VISIBLE_DEVICES={args.cuda_visible_devices})"

    print("Naive DDP benchmark setting")
    print(f"model_size={args.model_size} config={config} parameters={num_params:,}")
    print(f"world_size={args.world_size} device={device_label} backend={backend}")
    print(f"global_batch_size={args.global_batch_size} local_batch_size={local_batch_size} context_length={args.context_length} vocab_size={args.vocab_size}")
    print(f"optimizer={args.optimizer} amp_dtype={args.amp_dtype} warmup_steps={args.warmup_steps} measurement_steps={args.measurement_steps}")
    print()
    print(f"{'iter':>6} {'total max ms':>14} {'comm max ms':>14} {'comm %':>8} {'total mean ms':>14} {'comm mean ms':>14}")
    print("-" * 82)
    for summary in summaries:
        print(
            f"{summary.step:6d} "
            f"{summary.max_total_ms:14.3f} "
            f"{summary.max_communication_ms:14.3f} "
            f"{summary.max_communication_fraction * 100:8.2f} "
            f"{summary.mean_total_ms:14.3f} "
            f"{summary.mean_communication_ms:14.3f}"
        )


def write_csv(path: Path, summaries: list[StepTimingSummary]) -> None:
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
    validate_args(args) # 这会检查参数的有效性，并在发现问题时抛出异常。
    backend = resolve_backend(args) # 选择一种后端通信的方式（如 NCCL 或 Gloo），并返回后端的名称。
    num_params, summaries = run_benchmark(args, backend)
    print_results(args, backend, num_params, summaries)
    if args.csv is not None:
        write_csv(args.csv, summaries)
        print(f"\nWrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
