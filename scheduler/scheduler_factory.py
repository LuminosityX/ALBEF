""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler
from .plateau_lr import PlateauLRScheduler

# schedular: {sched: cosine, lr: 1e-4, epochs: 30, min_lr: 1e-5, decay_rate: 1, warmup_lr: 1e-5, warmup_epochs: 20, cooldown_epochs: 0}
def create_scheduler(args, optimizer):                 
    num_epochs = args.epochs                                       # 30

    if getattr(args, 'lr_noise', None) is not None:                # False
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    lr_scheduler = None
    if args.sched == 'cosine':                                     # True
        lr_scheduler = CosineLRScheduler(
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
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs           # 30 + 0
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    elif args.sched == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=args.min_lr,
            mode=mode,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            cooldown_t=0,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )

    return lr_scheduler, num_epochs                   # CosineLRScheduler, 30
