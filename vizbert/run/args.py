from pathlib import Path
from typing import Sequence
import argparse
import enum

import torch


__all__ = ['ArgumentParserBuilder', 'opt', 'OptionEnum']


def _make_parser_setter(option, key):
    def fn(value):
        option.kwargs[key] = value
        return option
    return fn


class ArgumentParserOption(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((self.args, self.kwargs))

    def __getattr__(self, item):
        if item == 'kwargs':
            return self.kwargs
        if item == 'args':
            return self.args
        return _make_parser_setter(self, item)


opt = ArgumentParserOption


class ArgumentParserBuilder(object):

    def __init__(self, **init_kwargs):
        self.parser = argparse.ArgumentParser(**init_kwargs)

    def add_opts(self, *options):
        for args, kwargs in options:
            self.parser.add_argument(*args, **kwargs)
        return self.parser


class OptionEnum(enum.Enum):
    DATA_FOLDER = opt('--data-folder', '-df', type=Path, required=True)
    WORKSPACE = opt('--workspace', '-w', type=Path, required=True)
    LAYER_IDX = opt('--layer-idx', '-l', type=int, required=True)
    PROBE_RANK = opt('--probe-rank', type=int, default=2)
    MODEL = opt('--model', default='bert-base-cased', type=str)
    DEVICE = opt('--device', type=torch.device, default='cuda:0')
    LR = opt('--lr', type=float, default=5e-4)
    NUM_WORKERS = opt('--num-workers', type=int, default=None)
    NUM_EPOCHS = opt('--num-epochs', type=int, default=1)
    BATCH_SIZE = opt('--batch-size', '-bsz', type=int, default=16)