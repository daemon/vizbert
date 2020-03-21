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

    def __init__(self, model: nn.Module, hooks: Sequence[InjectionHook], uninject=False):
        self.model = model
        self.hooks = hooks
        self.injected = uninject
        self.do_uninject = uninject

    def uninject(self):
        return ModelInjector(self.model, self.hooks, uninject=True)

    def __enter__(self, *args, **kwargs):
        if self.injected and not self.do_uninject:
            raise AlreadyInjectedError
        if not self.injected and self.do_uninject:
            raise NotInjectedError
        self.injected = not self.do_uninject
        for hook in self.hooks:
            if self.do_uninject:
                hook.eject(self.model)
            else:
                hook.inject(self.model)

    def __exit__(self, *args):
        if not self.injected and not self.do_uninject:
            raise NotInjectedError
        if self.injected and self.do_uninject:
            raise AlreadyInjectedError
        self.injected = not self.injected
        for hook in self.hooks[::-1]:
            if self.do_uninject:
                hook.inject(self.model)
            else:
                hook.eject(self.model)
