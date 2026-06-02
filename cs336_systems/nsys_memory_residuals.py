from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
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
    parser = argparse.ArgumentParser(description="Nsight Systems memory-lifecycle script for TransformerBlock residuals and gradients.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--amp-dtype", choices=sorted(AMP_DTYPES), default="none")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--target-block", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("memory_residual_reports"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("basics-adamw", "torch-adamw"), default="basics-adamw")
    parser.add_argument("--no-custom-nvtx", action="store_true", help="Disable custom NVTX ranges emitted by this script.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight memory profiling, but torch.cuda.is_available() is False.")
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required for Nsight memory profiling, but --device was {args.device!r}.")
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be at least 1 to initialize CUDA and optimizer state before profiling.")
    num_layers = MODEL_CONFIGS[args.model_size].num_layers
    if not 0 <= args.target_block < num_layers:
        raise ValueError(f"--target-block must be in [0, {num_layers - 1}] for model_size={args.model_size}.")


def amp_context(device: torch.device, amp_dtype: str):
    dtype = AMP_DTYPES[amp_dtype]
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


class Nvtx:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def push(self, name: str) -> None:
        if self.enabled:
            torch.cuda.nvtx.range_push(name)

    def pop(self) -> None:
        if self.enabled:
            torch.cuda.nvtx.range_pop()

    def range(self, name: str):
        return NvtxRange(self, name)


class NvtxRange:
    def __init__(self, nvtx: Nvtx, name: str):
        self.nvtx = nvtx
        self.name = name

    def __enter__(self):
        self.nvtx.push(self.name)

    def __exit__(self, exc_type, exc_value, traceback_obj):
        self.nvtx.pop()


class BlockInstrumentation:
    def __init__(self, model: BasicsTransformerLM, nvtx: Nvtx):
        self.model = model
        self.nvtx = nvtx
        self.block_stack: list[int] = []
        self.handles = []
        self.saved_records: list[dict[str, Any]] = []
        self.backward_memory_events: list[dict[str, Any]] = []

    def install(self) -> None:
        for block_idx, block in enumerate(self.model.layers):
            self.handles.append(block.register_forward_pre_hook(self._make_forward_pre_hook(block_idx)))
            self.handles.append(block.register_forward_hook(self._make_forward_hook(block_idx)))
            self.handles.append(block.register_full_backward_pre_hook(self._make_backward_pre_hook(block_idx)))
            self.handles.append(block.register_full_backward_hook(self._make_backward_hook(block_idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_forward_pre_hook(self, block_idx: int):
        def hook(module, inputs):
            self.block_stack.append(block_idx)
            self.nvtx.push(f"block/{block_idx:02d}/forward")

        return hook

    def _make_forward_hook(self, block_idx: int):
        def hook(module, inputs, output):
            self.nvtx.pop()
            popped = self.block_stack.pop()
            assert popped == block_idx

        return hook

    def _make_backward_pre_hook(self, block_idx: int):
        def hook(module, grad_output):
            self.backward_memory_events.append(
                {
                    "block": block_idx,
                    "event": "backward_pre",
                    "memory_allocated_bytes": torch.cuda.memory_allocated(),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(),
                }
            )
            self.nvtx.push(f"block/{block_idx:02d}/backward")

        return hook

    def _make_backward_hook(self, block_idx: int):
        def hook(module, grad_input, grad_output):
            self.backward_memory_events.append(
                {
                    "block": block_idx,
                    "event": "backward_post",
                    "memory_allocated_bytes": torch.cuda.memory_allocated(),
                    "memory_reserved_bytes": torch.cuda.memory_reserved(),
                }
            )
            self.nvtx.pop()

        return hook

    def pack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.block_stack and tensor.is_cuda:
            self.saved_records.append(
                {
                    "block": self.block_stack[-1],
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "logical_bytes": tensor.numel() * tensor.element_size(),
                    "storage_bytes": tensor.untyped_storage().nbytes(),
                    "storage_ptr": tensor.untyped_storage().data_ptr(),
                    "source": source_label_from_stack(),
                }
            )
        return tensor

    def unpack_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def source_label_from_stack() -> str:
    frames = traceback.extract_stack(limit=32)
    for frame in reversed(frames):
        filename = frame.filename.replace("\\", "/")
        if "cs336_basics" in filename or "cs336-basics" in filename:
            code = (frame.line or "").strip()
            return f"{Path(filename).name}:{frame.lineno}:{frame.name}:{code}"
    for frame in reversed(frames):
        filename = frame.filename.replace("\\", "/")
        if "site-packages/torch" not in filename and "torch/" not in filename:
            code = (frame.line or "").strip()
            return f"{Path(filename).name}:{frame.lineno}:{frame.name}:{code}"
    return "unknown"


def build_model(args: argparse.Namespace, config: ModelConfig, device: torch.device) -> BasicsTransformerLM:
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


def train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: str,
    nvtx: Nvtx,
) -> float:
    model.train()
    with nvtx.range("step/zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    with nvtx.range("step/forward"):
        with amp_context(device, amp_dtype):
            logits = model(input_ids)
    with nvtx.range("step/loss"):
        with amp_context(device, amp_dtype):
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    with nvtx.range("step/backward"):
        loss.backward()
    with nvtx.range("step/optimizer_step"):
        optimizer.step()
    return float(loss.detach().item())


def parameter_storage_ptrs(model: torch.nn.Module) -> set[int]:
    return {param.untyped_storage().data_ptr() for param in model.parameters() if param.is_cuda}


def summarize_saved_tensors(records: list[dict[str, Any]], target_block: int, param_ptrs: set[int]) -> dict[str, Any]:
    block_records = [record for record in records if record["block"] == target_block and record["storage_ptr"] not in param_ptrs]
    total_logical_bytes = sum(record["logical_bytes"] for record in block_records)
    source_bytes: dict[str, int] = defaultdict(int)
    for record in block_records:
        source_bytes[record["source"]] += record["logical_bytes"]

    top_sources = []
    for source, num_bytes in sorted(source_bytes.items(), key=lambda item: item[1], reverse=True)[:5]:
        top_sources.append(
            {
                "source": source,
                "bytes": num_bytes,
                "mib": bytes_to_mib(num_bytes),
                "percent": (100.0 * num_bytes / total_logical_bytes) if total_logical_bytes else 0.0,
            }
        )

    unique_storage: dict[int, int] = {}
    for record in block_records:
        storage_ptr = record["storage_ptr"]
        unique_storage[storage_ptr] = max(unique_storage.get(storage_ptr, 0), record["storage_bytes"])
    unique_storage_bytes = sum(unique_storage.values())

    return {
        "target_block": target_block,
        "num_saved_tensor_records": len(block_records),
        "saved_tensor_logical_bytes": total_logical_bytes,
        "saved_tensor_logical_mib": bytes_to_mib(total_logical_bytes),
        "unique_saved_storage_bytes": unique_storage_bytes,
        "unique_saved_storage_mib": bytes_to_mib(unique_storage_bytes),
        "top_5_sources_by_logical_saved_bytes": top_sources,
    }


def summarize_block_gradients(model: BasicsTransformerLM, target_block: int) -> dict[str, Any]:
    block = model.layers[target_block]
    grad_bytes_by_param = []
    total_grad_bytes = 0
    total_param_bytes = 0
    for name, param in block.named_parameters():
        param_bytes = param.numel() * param.element_size()
        grad_bytes = 0 if param.grad is None else param.grad.numel() * param.grad.element_size()
        total_param_bytes += param_bytes
        total_grad_bytes += grad_bytes
        grad_bytes_by_param.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "param_bytes": param_bytes,
                "grad_bytes": grad_bytes,
                "grad_mib": bytes_to_mib(grad_bytes),
            }
        )
    return {
        "target_block": target_block,
        "parameter_bytes": total_param_bytes,
        "parameter_mib": bytes_to_mib(total_param_bytes),
        "gradient_bytes": total_grad_bytes,
        "gradient_mib": bytes_to_mib(total_grad_bytes),
        "gradients_by_parameter": grad_bytes_by_param,
    }


def summarize_backward_memory_events(events: list[dict[str, Any]], target_block: int) -> dict[str, Any]:
    block_events = [event for event in events if event["block"] == target_block]
    pre = next((event for event in block_events if event["event"] == "backward_pre"), None)
    post = next((event for event in block_events if event["event"] == "backward_post"), None)
    if pre is None or post is None:
        return {"target_block": target_block, "available": False}
    allocated_delta = post["memory_allocated_bytes"] - pre["memory_allocated_bytes"]
    reserved_delta = post["memory_reserved_bytes"] - pre["memory_reserved_bytes"]
    return {
        "target_block": target_block,
        "available": True,
        "backward_pre_allocated_bytes": pre["memory_allocated_bytes"],
        "backward_post_allocated_bytes": post["memory_allocated_bytes"],
        "allocated_delta_bytes": allocated_delta,
        "allocated_delta_mib": bytes_to_mib(allocated_delta),
        "reserved_delta_bytes": reserved_delta,
        "reserved_delta_mib": bytes_to_mib(reserved_delta),
    }


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / 1024**2


def report_path(args: argparse.Namespace) -> Path:
    precision = "fp32" if args.amp_dtype == "none" else args.amp_dtype
    filename = f"residual_report_{args.model_size}_ctx{args.context_length}_bs{args.batch_size}_{precision}_block{args.target_block:02d}.json"
    return args.output_dir / filename


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def print_human_summary(report: dict[str, Any]) -> None:
    saved = report["saved_tensors"]
    gradients = report["gradients"]
    backward = report["backward_memory_events"]
    print(f"report_path={report['report_path']}")
    print(f"model_size={report['model_size']} context_length={report['context_length']} batch_size={report['batch_size']} precision={report['precision']}")
    print(f"target_block={saved['target_block']}")
    print(f"loss={report['loss']:.6f}")
    print(f"saved_tensor_logical={saved['saved_tensor_logical_mib']:.2f} MiB")
    print(f"unique_saved_storage={saved['unique_saved_storage_mib']:.2f} MiB")
    print(f"block_gradient_tensors={gradients['gradient_mib']:.2f} MiB")
    if backward["available"]:
        print(f"target_block_backward_allocated_delta={backward['allocated_delta_mib']:.2f} MiB")
    print("top_5_saved_tensor_sources:")
    for row in saved["top_5_sources_by_logical_saved_bytes"]:
        print(f"  {row['percent']:6.2f}% {row['mib']:10.2f} MiB  {row['source']}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    validate_args(args, device)

    config = MODEL_CONFIGS[args.model_size]
    nvtx = Nvtx(enabled=not args.no_custom_nvtx)
    model = build_model(args, config, device)
    optimizer = build_optimizer(args, model)
    input_ids, targets = make_batch(args, device)

    for _ in range(args.warmup_steps):
        train_step(model=model, optimizer=optimizer, input_ids=input_ids, targets=targets, device=device, amp_dtype=args.amp_dtype, nvtx=Nvtx(enabled=False))
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    instrumentation = BlockInstrumentation(model, nvtx)
    instrumentation.install()
    param_ptrs = parameter_storage_ptrs(model)

    try:
        with torch.autograd.graph.saved_tensors_hooks(instrumentation.pack_hook, instrumentation.unpack_hook):
            with nvtx.range("profile/train_step"):
                loss = train_step(model=model, optimizer=optimizer, input_ids=input_ids, targets=targets, device=device, amp_dtype=args.amp_dtype, nvtx=nvtx)
                torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA OOM while running model_size={args.model_size}, context_length={args.context_length}, "
            f"batch_size={args.batch_size}, precision={'fp32' if args.amp_dtype == 'none' else args.amp_dtype}. Try reducing --batch-size."
        ) from exc
    finally:
        instrumentation.remove()

    saved_summary = summarize_saved_tensors(instrumentation.saved_records, args.target_block, param_ptrs)
    gradient_summary = summarize_block_gradients(model, args.target_block)
    backward_summary = summarize_backward_memory_events(instrumentation.backward_memory_events, args.target_block)
    precision = "fp32" if args.amp_dtype == "none" else args.amp_dtype
    path = report_path(args)
    report = {
        "report_path": str(path),
        "model_size": args.model_size,
        "config": asdict(config),
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "precision": precision,
        "amp_dtype": args.amp_dtype,
        "target_block": args.target_block,
        "loss": loss,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_allocated_mib": bytes_to_mib(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "max_memory_reserved_mib": bytes_to_mib(torch.cuda.max_memory_reserved()),
        "saved_tensors": saved_summary,
        "gradients": gradient_summary,
        "backward_memory_events": backward_summary,
    }
    write_json(path, report)
    print_human_summary(report)


if __name__ == "__main__":
    main()
