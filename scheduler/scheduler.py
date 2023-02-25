from typing import Dict, Any

import torch


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.
    
    一个调度器基类，可用于调度任何优化器参数组。
    
    与内置的PyTorch调度器不同，该调度器的目的是为了持续调用
    * 在每个纪元结束时，在增加纪元计数之前，计算下一个纪元的值
    * 在每次优化器更新结束时，在增加更新次数后，计算下一次更新的值。
    ???
    
    建立在这个基础上的调度器应该尽量保持无状态（为了简单起见）。
    
    这个系列的调度器试图避免 "last_epoch "的含义和-1值的特殊行为的混淆。
    所有的纪元和更新次数都必须在训练代码中跟踪，并在相应的步骤或step_update调用中明确传递给调度器。

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,                  # 'lr'
                 noise_range_t=None,                      # None
                 noise_type='normal',                     # 'normal'
                 noise_pct=0.67,                          # 0.67
                 noise_std=1.0,                           # 1.0
                 noise_seed=None,                         # 42
                 initialize: bool = True) -> None:        # True
        self.optimizer = optimizer
        self.param_group_field = param_group_field                                                        # 'lr'
        self._initial_param_group_field = f"initial_{param_group_field}"                                  # 'initial_lr'
        if initialize:                                                                                    # True
            for i, group in enumerate(self.optimizer.param_groups):                                       # 该optimizer有两个groups
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])               # 字典中添加键值，这里添加了'initial_lr',值与'lr'相同
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups] # 有两个值，应该都是1e-4
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t                                                               # None
        self.noise_pct = noise_pct                                                                       # 0.67
        self.noise_type = noise_type                                                                     # 'normal'
        self.noise_std = noise_std                                                                       # 1.0
        self.noise_seed = noise_seed if noise_seed is not None else 42                                   # 42
        self.update_groups(self.base_values)                                                             # 将optimizer中的'lr'更新一下，这里就是设为最初的'initial_lr'

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs
