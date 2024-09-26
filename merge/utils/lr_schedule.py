import math

from torch.optim.lr_scheduler import LRScheduler


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr
                + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
