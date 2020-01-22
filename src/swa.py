import argus
from torchcontrib.optim import SWA

from src.utils import initialize_amp


class SWACallback(argus.callbacks.Callback):
    def __init__(self, swa_start=None, swa_freq=None,
                 swa_lr=None, swa_save_path="",
                 use_amp=False):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_save_path = swa_save_path
        self.model = None
        self.use_amp = use_amp

    def start(self, state: argus.engine.State):
        self.model = state.model
        self.model.optimizer = SWA(self.model.optimizer,
                                   swa_start=self.swa_start,
                                   swa_freq=self.swa_freq,
                                   swa_lr=self.swa_lr)
        if self.use_amp:
            initialize_amp(self.model)

    def complete(self, state: argus.engine.State):
        self.model.optimizer.swap_swa_sgd()
        if self.swa_save_path:
            self.model.save(self.swa_save_path)
