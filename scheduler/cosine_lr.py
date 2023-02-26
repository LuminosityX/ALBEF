""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import math
import numpy as np
import torch

from .scheduler import Scheduler

from pdb import set_trace as breakpoint

_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """
    
    '''
            optimizer,
            t_initial=num_epochs,                                  # 30 
            t_mul=getattr(args, 'lr_cycle_mul', 1.),               # 1.
            lr_min=args.min_lr,                                    # 1e-5
            decay_rate=args.decay_rate,                            # 1
            warmup_lr_init=args.warmup_lr,                         # 1e-5
            warmup_t=args.warmup_epochs,                           # 20
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),        # 1
            t_in_epochs=True,
            noise_range_t=noise_range,                             # None
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        
    '''

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,                                  # 30
                 t_mul: float = 1.,                               # 1.
                 lr_min: float = 0.,                              # 1e-5
                 decay_rate: float = 1.,                          # 1
                 warmup_t=0,                                      # 20
                 warmup_lr_init=0,                                # 1e-5
                 warmup_prefix=True,                              # True
                 cycle_limit=0,                                   # 1
                 t_in_epochs=True,                                # True
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial                                                                      # 30  
        self.t_mul = t_mul                                                                              # 1.
        self.lr_min = lr_min                                                                            # 1e-5
        self.decay_rate = decay_rate                                                                    # 1
        self.cycle_limit = cycle_limit                                                                  # 1
        self.warmup_t = warmup_t                                                                        # 20
        self.warmup_lr_init = warmup_lr_init                                                            # 1e-5
        self.warmup_prefix = warmup_prefix                                                              # True
        self.t_in_epochs = t_in_epochs                                                                  # True
        if self.warmup_t:                                                                               # 20
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]        # (1e-4 - 1e-5)/20
            super().update_groups(self.warmup_lr_init)                                                  # 将optimizer的'lr'设为了self.warmup_lr_init，即1e-5
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]                              # 当在warm_up过程中时，计算出的学习率
        else:
            if self.warmup_prefix:                                                                      # True
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial                                                                 # ??? 这里不是等于0吗
                t_i = self.t_initial                                                                    # 30
                t_curr = t - (self.t_initial * i)                                                       # 就还是t

            gamma = self.decay_rate ** i                                                                # 1的0次方，还是1
            lr_min = self.lr_min * gamma                                                                # 1e-5
            lr_max_values = [v * gamma for v in self.base_values]                                       # []两个1e-4

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):                # 满足后面的
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:                                                            # True
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):                                            # create_scheduler中的调用
        if not cycles:                                                               # not 0, 应该是True
            cycles = self.cycle_limit                                                # 1
        cycles = max(1, cycles)
        if self.t_mul == 1.0:                                                        # True
            return self.t_initial * cycles                                           # 30*1 
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))
