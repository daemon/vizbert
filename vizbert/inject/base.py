from typing import Sequence

import torch.nn as nn


class ForwardWrapper(nn.Module):

    def __init__(self, module: nn.Module, fn):
        super().__init__()
        self.module = module
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(self.module(*args, **kwargs))


class InjectionHook(object):

    def do_inject(self, model: nn.Module):
        raise NotImplementedError

    def do_eject(self, model: nn.Module):
        raise NotImplementedError

    def inject(self, model: nn.Module):
        if hasattr(model, '_injection_hooks') and self.name in model._injection_hooks:
            raise AlreadyInjectedError
        self.do_inject(model)
        if not hasattr(model, '_injection_hooks'):
            setattr(model, '_injection_hooks', set())
        model._injection_hooks.add(self.name)

    def eject(self, model: nn.Module):
        if hasattr(model, '_injection_hooks') and not self.name in model._injection_hooks:
            raise NotInjectedError
        self.do_eject(model)
        model._injection_hooks.remove(self.name)

    @property
    def name(self):
        raise NotImplementedError


class AlreadyInjectedError(Exception):
    pass


class NotInjectedError(Exception):
    pass


class ModelInjector(object):

    def __init__(self, model: nn.Module, hooks: Sequence[InjectionHook]):
        self.model = model
        self.hooks = hooks

    def __enter__(self, *args, **kwargs):
        for hook in self.hooks:
            hook.inject(self.model)

    def __exit__(self, *args):
        for hook in self.hooks[::-1]:
            hook.eject(self.model)
