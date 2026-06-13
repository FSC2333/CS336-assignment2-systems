from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class _ShardedParamInfo:
    name: str
    module: nn.Module
    parameter: nn.Parameter
    full_shape: torch.Size
    shard_start: int
    shard_length: int
    shard_lengths: list[int]
    max_shard_length: int
    local_shard: torch.Tensor | None = None
    gathered: bool = False


@dataclass
class _PendingShardedGrad:
    info: _ShardedParamInfo
    handle: dist.Work
    buffer: torch.Tensor
    used_reduce_scatter: bool
    input_buffer: torch.Tensor | None = None


@dataclass
class _PendingReplicatedGrad:
    parameter: nn.Parameter
    handle: dist.Work


def _target_module_types() -> tuple[type[nn.Module], ...]:
    types: list[type[nn.Module]] = [nn.Linear, nn.Embedding]
    try:
        from cs336_basics.model import Embedding, Linear

        types.extend([Linear, Embedding])
    except Exception:
        pass
    return tuple(types)


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype

        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        self._target_types = _target_module_types()
        self._module_infos: dict[nn.Module, list[_ShardedParamInfo]] = {}
        self._name_to_info: dict[str, _ShardedParamInfo] = {}
        self._param_id_to_info: dict[int, _ShardedParamInfo] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pending_sharded_grads: list[_PendingShardedGrad] = []
        self._pending_replicated_grads: list[_PendingReplicatedGrad] = []
        self._devices_to_synchronize: set[torch.device] = set()

        self._broadcast_module_state()
        self._shard_target_module_parameters()
        self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for pending in self._pending_sharded_grads:
            pending.handle.wait()
            if pending.used_reduce_scatter:
                grad = pending.buffer
            else:
                grad = self._slice_local_shard(pending.buffer, pending.info).clone().contiguous()
            grad.div_(self.world_size)
            pending.info.parameter.grad = grad.to(dtype=pending.info.parameter.data.dtype)
            if grad.device.type == "cuda":
                self._devices_to_synchronize.add(grad.device)
        self._pending_sharded_grads.clear()

        for pending in self._pending_replicated_grads:
            pending.handle.wait()
            if pending.parameter.grad is not None:
                pending.parameter.grad.div_(self.world_size)
                if pending.parameter.grad.device.type == "cuda":
                    self._devices_to_synchronize.add(pending.parameter.grad.device)
        self._pending_replicated_grads.clear()

        for device in self._devices_to_synchronize:
            torch.cuda.synchronize(device)
        self._devices_to_synchronize.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.module.named_parameters():
            info = self._name_to_info.get(name)
            if info is None:
                state[name] = parameter.detach().clone()
            else:
                state[name] = self._gather_full_tensor(info, dtype=self._master_dtype(info))
        return state

    def _broadcast_module_state(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    @torch.no_grad()
    def _shard_target_module_parameters(self) -> None:
        module_names = dict(self.module.named_modules())
        for module_name, submodule in module_names.items():
            if not isinstance(submodule, self._target_types):
                continue

            infos: list[_ShardedParamInfo] = []
            for param_name, parameter in submodule.named_parameters(recurse=False):
                if id(parameter) in self._param_id_to_info:
                    info = self._param_id_to_info[id(parameter)]
                    infos.append(info)
                    continue
                if parameter.ndim == 0:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                full_data = parameter.detach()
                if full_data.is_floating_point():
                    full_data = full_data.to(torch.float32)

                shard_lengths = self._split_lengths(full_data.shape[0])
                shard_start = sum(shard_lengths[: self.rank])
                shard_length = shard_lengths[self.rank]
                local_shard = full_data.narrow(0, shard_start, shard_length).clone().contiguous()
                parameter.data = local_shard

                info = _ShardedParamInfo(
                    name=full_name,
                    module=submodule,
                    parameter=parameter,
                    full_shape=torch.Size(full_data.shape),
                    shard_start=shard_start,
                    shard_length=shard_length,
                    shard_lengths=shard_lengths,
                    max_shard_length=max(shard_lengths),
                    local_shard=parameter.data,
                )
                infos.append(info)
                self._name_to_info[full_name] = info
                self._param_id_to_info[id(parameter)] = info

            if infos:
                self._module_infos[submodule] = infos

    def _register_hooks(self) -> None:
        for module, infos in self._module_infos.items():
            self._hook_handles.append(module.register_forward_pre_hook(self._make_forward_pre_hook(infos)))
            self._hook_handles.append(module.register_forward_hook(self._make_forward_post_hook(infos)))
            if any(not info.parameter.requires_grad for info in infos):
                self._hook_handles.append(module.register_full_backward_hook(self._make_backward_post_hook(infos)))

        sharded_param_ids = set(self._param_id_to_info)
        for info in self._name_to_info.values():
            if info.parameter.requires_grad:
                self._hook_handles.append(info.parameter.register_post_accumulate_grad_hook(self._make_sharded_grad_hook(info)))

        for parameter in self.module.parameters():
            if parameter.requires_grad and id(parameter) not in sharded_param_ids:
                self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._make_replicated_grad_hook()))

    def _make_forward_pre_hook(self, infos: list[_ShardedParamInfo]):
        @torch.no_grad()
        def hook(module: nn.Module, inputs) -> None:
            for info in infos:
                self._all_gather_param(info)

        return hook

    def _make_forward_post_hook(self, infos: list[_ShardedParamInfo]):
        @torch.no_grad()
        def hook(module: nn.Module, inputs, output) -> None:
            self._register_backward_gather_hook(output, infos)
            for info in infos:
                self._release_param(info)

        return hook

    def _make_backward_post_hook(self, infos: list[_ShardedParamInfo]):
        @torch.no_grad()
        def hook(module: nn.Module, grad_input, grad_output) -> None:
            for info in infos:
                if not info.parameter.requires_grad:
                    self._release_param(info)

        return hook

    def _register_backward_gather_hook(self, output, infos: list[_ShardedParamInfo]) -> None:
        has_gathered = False

        def gather_before_backward(grad: torch.Tensor) -> torch.Tensor:
            nonlocal has_gathered
            if not has_gathered:
                with torch.no_grad():
                    for info in infos:
                        self._all_gather_param(info)
                has_gathered = True
            return grad

        def register_on_tensors(value) -> None:
            if torch.is_tensor(value):
                if value.requires_grad:
                    value.register_hook(gather_before_backward)
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    register_on_tensors(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    register_on_tensors(item)

        register_on_tensors(output)

    def _make_sharded_grad_hook(self, info: _ShardedParamInfo):
        @torch.no_grad()
        def hook(parameter: nn.Parameter) -> None:
            if parameter.grad is None:
                self._release_param(info)
                return

            full_grad = parameter.grad.detach()
            if full_grad.is_floating_point():
                full_grad = full_grad.to(torch.float32)
            full_grad = full_grad.contiguous()

            self._release_param(info)

            if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
                parameter.grad = self._slice_local_shard(full_grad, info).clone().contiguous().to(dtype=parameter.data.dtype)
                return

            parameter.grad = None
            pending = self._start_sharded_gradient_sync(info, full_grad)
            self._pending_sharded_grads.append(pending)

        return hook

    def _make_replicated_grad_hook(self):
        @torch.no_grad()
        def hook(parameter: nn.Parameter) -> None:
            if parameter.grad is None:
                return
            if parameter.grad.dtype != parameter.data.dtype:
                parameter.grad = parameter.grad.to(dtype=parameter.data.dtype)

            if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
                return

            handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_replicated_grads.append(_PendingReplicatedGrad(parameter=parameter, handle=handle))

        return hook

    @torch.no_grad()
    def _all_gather_param(self, info: _ShardedParamInfo) -> None:
        if info.gathered:
            return
        info.local_shard = info.parameter.data
        full_param = self._gather_full_tensor(info, dtype=self._compute_dtype(info))
        info.parameter.data = full_param
        info.gathered = True

    @torch.no_grad()
    def _release_param(self, info: _ShardedParamInfo) -> None:
        if not info.gathered:
            return
        if info.local_shard is not None:
            info.parameter.data = info.local_shard
        info.gathered = False

    def _start_sharded_gradient_sync(self, info: _ShardedParamInfo, full_grad: torch.Tensor) -> _PendingShardedGrad:
        if self._can_reduce_scatter(info):
            output = torch.empty_like(self._slice_local_shard(full_grad, info))
            try:
                handle = dist.reduce_scatter_tensor(output, full_grad, op=dist.ReduceOp.SUM, async_op=True)
                return _PendingShardedGrad(info=info, handle=handle, buffer=output, used_reduce_scatter=True, input_buffer=full_grad)
            except RuntimeError:
                pass

        buffer = full_grad.clone().contiguous()
        handle = dist.all_reduce(buffer, op=dist.ReduceOp.SUM, async_op=True)
        return _PendingShardedGrad(info=info, handle=handle, buffer=buffer, used_reduce_scatter=False)

    def _gather_full_tensor(self, info: _ShardedParamInfo, dtype: torch.dtype) -> torch.Tensor:
        local = info.local_shard if info.gathered and info.local_shard is not None else info.parameter.data
        local = local.detach()
        if local.dtype != dtype:
            local = local.to(dtype=dtype)

        if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
            return local.clone().contiguous()

        padded = self._pad_to_max_shard(local, info)
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded.contiguous())
        chunks = [tensor.narrow(0, 0, length) for tensor, length in zip(gathered, info.shard_lengths, strict=True)]
        return torch.cat(chunks, dim=0).contiguous()

    def _pad_to_max_shard(self, tensor: torch.Tensor, info: _ShardedParamInfo) -> torch.Tensor:
        if info.shard_length == info.max_shard_length:
            return tensor
        padded_shape = (info.max_shard_length, *tensor.shape[1:])
        padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
        if info.shard_length > 0:
            padded.narrow(0, 0, info.shard_length).copy_(tensor)
        return padded

    def _slice_local_shard(self, tensor: torch.Tensor, info: _ShardedParamInfo) -> torch.Tensor:
        return tensor.narrow(0, info.shard_start, info.shard_length)

    def _split_lengths(self, size: int) -> list[int]:
        base = size // self.world_size
        remainder = size % self.world_size
        return [base + (rank < remainder) for rank in range(self.world_size)]

    def _can_reduce_scatter(self, info: _ShardedParamInfo) -> bool:
        return all(length == info.shard_length for length in info.shard_lengths) and hasattr(dist, "reduce_scatter_tensor")

    def _compute_dtype(self, info: _ShardedParamInfo) -> torch.dtype:
        if self.compute_dtype is not None and info.parameter.data.is_floating_point():
            return self.compute_dtype
        return info.parameter.data.dtype

    def _master_dtype(self, info: _ShardedParamInfo) -> torch.dtype:
        local = info.local_shard if info.gathered and info.local_shard is not None else info.parameter.data
        return local.dtype
