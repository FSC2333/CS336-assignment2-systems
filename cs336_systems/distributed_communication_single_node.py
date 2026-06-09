from __future__ import annotations

import argparse
import csv
import os
import queue
import socket
import statistics
import timeit
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BYTES_PER_FLOAT32 = torch.finfo(torch.float32).bits // 8
DEFAULT_SIZES_MIB = (1, 10, 100, 1024)
DEFAULT_WORLD_SIZES = (2, 4)
DEVICE_MODES = ("single", "rank")


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    device: str
    world_size: int
    size_label: str
    size_mib: int
    numel: int
    warmup_steps: int
    measurement_steps: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    mean_step_max_ms: float
    payload_gib_per_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-node multi-GPU distributed all-reduce latency for float32 tensors.")
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=list(DEFAULT_SIZES_MIB), help="Float32 tensor payload sizes in MiB. Defaults to 1, 10, 100, and 1024 MiB.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=list(DEFAULT_WORLD_SIZES), help="Process/GPU counts to benchmark. Defaults to 2 and 4.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto", help="Use NCCL for CUDA rank mode by default, otherwise Gloo.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Use CUDA by default when available.")
    parser.add_argument("--device-mode", choices=DEVICE_MODES, default="rank", help="rank maps rank i to cuda:i; single puts every rank on --cuda-device.")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index used by every rank when --device-mode single.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES mask set before CUDA initialization. Use 1 to restrict all ranks to physical GPU 1, where it appears as cuda:0.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=20)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--process-group-timeout-s", type=int, default=120)
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to write the benchmark table as CSV.")
    parser.add_argument("--strict-world-sizes", action="store_true", help="Fail instead of skipping world sizes that exceed the available CUDA device count.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(size_mib <= 0 for size_mib in args.sizes_mib):
        raise ValueError("--sizes-mib must contain positive integers.")
    if any(world_size <= 0 for world_size in args.world_sizes):
        raise ValueError("--world-sizes must contain positive integers.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.measurement_steps <= 0:
        raise ValueError("--measurement-steps must be positive.")
    if args.process_group_timeout_s <= 0:
        raise ValueError("--process-group-timeout-s must be positive.")
    if args.cuda_device < 0:
        raise ValueError("--cuda-device must be non-negative.")


def resolve_backend_and_device(args: argparse.Namespace) -> tuple[str, str]:
    cuda_available = torch.cuda.is_available()

    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device

    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    if args.backend == "auto":
        backend = "nccl" if device == "cuda" and args.device_mode == "rank" else "gloo"
    else:
        backend = args.backend

    if backend == "nccl" and device != "cuda":
        raise ValueError("The NCCL backend requires --device cuda.")

    return backend, device


def filter_world_sizes(world_sizes: list[int], device: str, strict: bool) -> list[int]:
    if device != "cuda":
        return world_sizes

    available_gpus = torch.cuda.device_count()
    valid_world_sizes = [world_size for world_size in world_sizes if world_size <= available_gpus]
    skipped_world_sizes = [world_size for world_size in world_sizes if world_size > available_gpus]
    if skipped_world_sizes:
        message = f"Skipping world sizes {skipped_world_sizes}; only {available_gpus} CUDA device(s) are visible."
        if strict:
            raise RuntimeError(message)
        print(message)
    return valid_world_sizes


def validate_cuda_device(args: argparse.Namespace, device: str) -> None:
    if device != "cuda":
        return

    available_gpus = torch.cuda.device_count()
    if args.cuda_device >= available_gpus:
        raise RuntimeError(f"--cuda-device {args.cuda_device} was requested, but only {available_gpus} CUDA device(s) are visible.")


def find_free_port(addr: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((addr, 0))
        return int(sock.getsockname()[1])


def size_label(size_mib: int) -> str:
    if size_mib % 1024 == 0:
        return f"{size_mib // 1024} GiB"
    return f"{size_mib} MiB"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup_process_group(*, backend: str, master_addr: str, master_port: int, rank: int, world_size: int, timeout_s: int) -> None:
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )

# 对固定大小的一个 tensor，测量多次 all_reduce 通信耗时，并返回当前 rank 测到的时间列表。
def benchmark_one_size(
    *,
    tensor: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measurement_steps: int,
) -> list[float]:
    for _ in range(warmup_steps):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    synchronize(device)
    dist.barrier() # 让所有 rank 在这里集合。只有所有 rank 都到达这一行，大家才会继续往下走。作用是保证：所有 rank 都完成 warm-up 后，再一起开始正式计时。
    synchronize(device)

    timings_s = []
    for _ in range(measurement_steps):
        dist.barrier()
        synchronize(device)
        start = timeit.default_timer()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
        synchronize(device)
        timings_s.append(timeit.default_timer() - start)

    dist.barrier()
    return timings_s


def summarize_timings(
    *,
    backend: str,
    device_name: str,
    world_size: int,
    size_mib: int,
    numel: int,
    warmup_steps: int,
    measurement_steps: int,
    rank_timings_s: list[list[float]],
) -> BenchmarkRow:
    flattened_ms = [timing_s * 1_000 for rank_timings in rank_timings_s for timing_s in rank_timings]
    per_step_max_ms = [max(rank_timings[step_idx] for rank_timings in rank_timings_s) * 1_000 for step_idx in range(measurement_steps)]
    mean_step_max_ms = statistics.fmean(per_step_max_ms)
    payload_gib = size_mib / 1024
    payload_gib_per_s = payload_gib / (mean_step_max_ms / 1_000)
    return BenchmarkRow(
        backend=backend,
        device=device_name,
        world_size=world_size,
        size_label=size_label(size_mib),
        size_mib=size_mib,
        numel=numel,
        warmup_steps=warmup_steps,
        measurement_steps=measurement_steps,
        mean_ms=statistics.fmean(flattened_ms),
        std_ms=statistics.stdev(flattened_ms) if len(flattened_ms) > 1 else 0.0,
        min_ms=min(flattened_ms),
        max_ms=max(flattened_ms),
        mean_step_max_ms=mean_step_max_ms,
        payload_gib_per_s=payload_gib_per_s,
    )


def benchmark_device_label(device_name: str, device_mode: str, cuda_device: int, world_size: int) -> str:
    if device_name != "cuda":
        return device_name
    if device_mode == "single":
        return f"cuda:{cuda_device}"
    return f"cuda:0-{world_size - 1}"


def worker(
    rank: int,
    world_size: int,
    backend: str,
    device_name: str,
    device_mode: str,
    cuda_device: int,
    sizes_mib: list[int],
    warmup_steps: int,
    measurement_steps: int,
    master_addr: str,
    master_port: int,
    process_group_timeout_s: int,
    result_queue: mp.SimpleQueue,
) -> None:
    if device_name == "cuda":
        device_index = cuda_device if device_mode == "single" else rank
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")

    setup_process_group(backend=backend, master_addr=master_addr, master_port=master_port, rank=rank, world_size=world_size, timeout_s=process_group_timeout_s)

    rows = []
    try:
        for size_mib in sizes_mib:
            if rank == 0:
                print(f"  benchmarking size={size_label(size_mib)}", flush=True)
            size_bytes = size_mib * 1024**2
            numel = size_bytes // BYTES_PER_FLOAT32 # 计算需要多少个 float32 元素。float32 每个元素 4 bytes，所以 1 MiB 对应 262144 个元素。
            tensor = torch.zeros(numel, dtype=torch.float32, device=device)
            synchronize(device) # 确保 tensor 创建等 CUDA 操作真的完成

            local_timings_s = benchmark_one_size(tensor=tensor, device=device, warmup_steps=warmup_steps, measurement_steps=measurement_steps) # 测试
            gathered_timings_s: list[list[float] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_timings_s, local_timings_s) # 每个 rank 把自己的 local_timings_s 发出去，同时也接收其他所有 rank 的 local_timings_s。

            if rank == 0:
                rank_timings_s = [timings for timings in gathered_timings_s if timings is not None]
                row_device_name = benchmark_device_label(device_name, device_mode, cuda_device, world_size)
                rows.append(
                    summarize_timings(
                        backend=backend,
                        device_name=row_device_name,
                        world_size=world_size,
                        size_mib=size_mib,
                        numel=numel,
                        warmup_steps=warmup_steps,
                        measurement_steps=measurement_steps,
                        rank_timings_s=rank_timings_s,
                    )
                )

            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()
            dist.barrier()

        if rank == 0:
            result_queue.put(rows)
    finally:
        dist.destroy_process_group()


def run_world_size(
    *,
    world_size: int,
    backend: str,
    device_name: str,
    device_mode: str,
    cuda_device: int,
    sizes_mib: list[int],
    warmup_steps: int,
    measurement_steps: int,
    master_addr: str,
    process_group_timeout_s: int,
) -> list[BenchmarkRow]:
    
    ctx = mp.get_context("spawn")
    # 获取 Python multiprocessing 的 "spawn" 启动方式。
    # 这对 CUDA 很重要：spawn 会为每个子进程启动一个干净的 Python 解释器，比 fork 更适合 CUDA/NCCL，避免继承父进程里的 CUDA 状态。
    result_queue = ctx.SimpleQueue()
    # 创建一个进程间通信队列。
    # 子进程里的 rank 0 会把汇总后的 benchmark 结果放进这个 queue，父进程最后从这里取结果。
    master_port = find_free_port(master_addr) # 后面创建的所有子进程都会用同一个地址：端口号，从而发现并组成同一个通信组
    mp.spawn( # 创建多个子进程，每个子进程运行 worker 函数，传入必要的参数
        worker,
        args=(
            world_size,
            backend,
            device_name,
            device_mode,
            cuda_device,
            sizes_mib,
            warmup_steps,
            measurement_steps,
            master_addr,
            master_port,
            process_group_timeout_s,
            result_queue,
        ),
        nprocs=world_size, # 启动子进程的数目
        join=True, #父进程等待所有子进程结束
    )
    return result_queue.get()


def write_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_results(rows: list[BenchmarkRow]) -> None:
    print()
    print(f"{'backend':<8} {'device':<6} {'ranks':>5} {'size':>8} {'mean ms':>10} {'rank-max ms':>12} {'std ms':>10} {'min ms':>10} {'max ms':>10} {'GiB/s':>10}")
    print("-" * 105)
    for row in rows:
        print(
            f"{row.backend:<8} {row.device:<6} {row.world_size:5d} {row.size_label:>8} "
            f"{row.mean_ms:10.3f} {row.mean_step_max_ms:12.3f} {row.std_ms:10.3f} {row.min_ms:10.3f} {row.max_ms:10.3f} {row.payload_gib_per_s:10.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    validate_args(args)
    backend, device_name = resolve_backend_and_device(args)
    validate_cuda_device(args, device_name)
    if args.device_mode == "rank":
        world_sizes = filter_world_sizes(args.world_sizes, device_name, args.strict_world_sizes)
    else:  # single
        world_sizes = args.world_sizes
    if not world_sizes:
        raise RuntimeError("No benchmarkable world sizes remain after filtering.")

    rows: list[BenchmarkRow] = []
    for world_size in world_sizes:
        device_label = benchmark_device_label(device_name, args.device_mode, args.cuda_device, world_size)
        print(f"Running all-reduce benchmark: backend={backend} device={device_label} device_mode={args.device_mode} world_size={world_size}", flush=True)
        rows.extend(
            run_world_size(
                world_size=world_size,
                backend=backend,
                device_name=device_name,
                device_mode=args.device_mode,
                cuda_device=args.cuda_device,
                sizes_mib=args.sizes_mib,
                warmup_steps=args.warmup_steps,
                measurement_steps=args.measurement_steps,
                master_addr=args.master_addr,
                process_group_timeout_s=args.process_group_timeout_s,
            )
        )

    print_results(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"\nWrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
