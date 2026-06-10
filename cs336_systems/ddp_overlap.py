from __future__ import annotations

import torch
import torch.distributed as dist


class OverlapDistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles = []  # 保存异步 all-reduce 返回的 handle
        self._hook_handles = []  # 保存注册的 hook 的 handle，以便在必要时移除它们
        self._devices_to_synchronize: set[torch.device] = set()  # 记录哪些 CUDA device 上发起过梯度通信。最后可以对这些 device 调用 torch.cuda.synchronize()。

        self._broadcast_module_state()  # 广播 rank0 的模型参数
        self._register_gradient_hooks()  # 定义注册梯度 hook 的函数。

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _broadcast_module_state(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _register_gradient_hooks(self) -> None:
        for parameter in self.module.parameters():
            if parameter.requires_grad:  # 只给训练主要的参数注册hook
                self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._make_gradient_hook()))
                # 注册回调函数，在每次梯度累积完成后被调用。这个 hook 会触发 all-reduce 来同步梯度，并且在通信完成之前不会阻塞后续的计算，从而实现计算和通信的重叠。

    def _make_gradient_hook(self):
        @torch.no_grad()  # hook 里面不需要被 autograd 追踪。我们会直接修改 parameter.grad，所以禁用梯度记录。
        def hook(parameter: torch.nn.Parameter) -> None:
            if not dist.is_available() or not dist.is_initialized():
                return

            world_size = dist.get_world_size()
            if world_size == 1 or parameter.grad is None:
                return

            parameter.grad.div_(world_size)
            handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

            if parameter.grad.device.type == "cuda":
                self._devices_to_synchronize.add(parameter.grad.device)

        return hook

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()  # 确保在使用优化器之前，所有的 all-reduce 都已经完成了。

        for device in self._devices_to_synchronize:
            torch.cuda.synchronize(device)  # 确保所有通信相关的 CUDA 操作都完成了，避免在 optimizer step 过程中出现未定义行为。
        self._devices_to_synchronize.clear()
