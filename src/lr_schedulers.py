from torch.optim import lr_scheduler as _scheduler

from argus.engine import State
from argus.callbacks.callback import Callback


class LRScheduler(Callback):
    def __init__(self, scheduler_factory, monitor=None):
        self.scheduler_factory = scheduler_factory
        self._monitor = monitor
        self._scheduler = None

    def start(self, state: State):
        if self._scheduler is None:
            self._scheduler = self.scheduler_factory(state.model.optimizer)

    def epoch_start(self, state: State):
        self._scheduler.step()


class CosineAnnealingLR(LRScheduler):
    def __init__(self, T_max, eta_min=0):
        super().__init__(lambda opt: _scheduler.CosineAnnealingLR(opt,
                                                                  T_max,
                                                                  eta_min=eta_min))
