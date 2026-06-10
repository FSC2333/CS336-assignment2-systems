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
        self._all_params: list[torch.nn.Parameter] = []
        self._param_owners: list[int] = []
        self._local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: Optimizer | None = None

        super().__init__(params, defaults=kwargs)

        if self._local_param_groups:
            self._local_optimizer = optimizer_cls(self._local_param_groups, **kwargs)
            self.state = self._local_optimizer.state

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]

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
