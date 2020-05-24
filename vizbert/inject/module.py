from torch import nn as nn

from vizbert.utils import Timer
from .base import InjectionHook


__all__ = ['InstrumentModuleInjectionHook']


class InstrumentModuleInjectionHook(InjectionHook):
    class TimerWrapper:
        def __init__(self, timer, name):
            self.timer, self.name = timer, name

        def __call__(self, *args, **kwargs):
            self.timer.enter(self.name)

    def __init__(self):
        self.timer = Timer()
        self.handles = []

    def do_inject(self, model: nn.Module):
        for name, module in model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self.TimerWrapper(self.timer, name)))
            self.handles.append(module.register_forward_hook(lambda *args, **kwargs: self.timer.exit()))

    def do_eject(self, model: nn.Module):
        for handle in self.handles:
            handle.remove()

    @property
    def name(self):
        return 'instrument-hook'
