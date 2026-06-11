from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self._all_params: list[torch.nn.Parameter] = [] # 保存所有参数，顺序是全局一致的。后面同步参数时会遍历这个列表
        self._param_owners: list[int] = [] # 保存每个参数的owner rank，顺序和_all_params一致。后面同步参数时会用到这个列表
        self._local_param_groups: list[dict[str, Any]] = [] # 保存当前 rank 真正负责优化的 parameter groups。底层 optimizer 只会拿到这些本地参数。
        self._local_optimizer: Optimizer | None = None

        super().__init__(params, defaults=kwargs)

        if self._local_param_groups:
            self._local_optimizer = optimizer_cls(self._local_param_groups, **kwargs) # 创建真正负责更新本 rank 参数的 optimizer。
            self.state = self._local_optimizer.state

    def add_param_group(self, param_group: dict[str, Any]) -> None: 
        # 重写 add_param_group。父类构造函数期间会调用它；训练过程中如果用户动态添加参数组，也会调用它。
        super().add_param_group(param_group)
        group = self.param_groups[-1] # 获取刚添加的参数组。注意父类的 add_param_group 已经把 param_group 添加到 self.param_groups 了。

        local_params = []
        for parameter in group["params"]:
            owner = len(self._all_params) % self.world_size
            self._all_params.append(parameter)
            self._param_owners.append(owner)
            if owner == self.rank:
                local_params.append(parameter)

        if not local_params:
            return

        local_group = {key: value for key, value in group.items() if key != "params"}
        local_group["params"] = local_params

        if self._local_optimizer is None:
            self._local_param_groups.append(local_group)
        else:
            self._local_optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        loss = None
        if self._local_optimizer is not None:
            if closure is None:
                loss = self._local_optimizer.step(**kwargs)
            else:
                loss = self._local_optimizer.step(closure=closure, **kwargs)

        self._synchronize_parameters()
        return loss

    def _synchronize_parameters(self) -> None:
        if not dist.is_available() or not dist.is_initialized() or self.world_size == 1:
            return

        for parameter, owner in zip(self._all_params, self._param_owners, strict=True):
            dist.broadcast(parameter.data, src=owner)
