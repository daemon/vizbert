from typing import Sequence

import torch.nn as nn


__all__ = ['ForwardWrapper',
           'InjectionHook',
           'AlreadyInjectedError',
           'NotInjectedError',
           'ModelInjector']


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
        self.do_inject(model)

    def eject(self, model: nn.Module):
        self.do_eject(model)

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
        self.injected = False

    def __enter__(self, *args, **kwargs):
        if self.injected:
            raise AlreadyInjectedError
        self.injected = True
        for hook in self.hooks:
            hook.inject(self.model)

    def __exit__(self, *args):
        if not self.injected:
            raise NotInjectedError
        self.injected = False
        for hook in self.hooks[::-1]:
            hook.eject(self.model)
