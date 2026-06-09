from __future__ import annotations

import torch
import torch.distributed as dist


# 初始化时：让所有 rank 的模型参数一致。
# backward 后：让所有 rank 的梯度一致。

class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_module_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None: #同步模型初始状态的辅助函数。
        if not dist.is_available() or not dist.is_initialized():
            return

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def finish_gradient_synchronization(self) -> None:  # 这是 backward 之后、optimizer step 之前调用的函数。它负责同步梯度。
        if not dist.is_available() or not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        devices_to_synchronize: set[torch.device] = set()
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue

            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size) # 得到平均梯度
            if parameter.grad.device.type == "cuda":
                devices_to_synchronize.add(parameter.grad.device)

        for device in devices_to_synchronize:
            torch.cuda.synchronize(device)
