from pathlib import Path
import argparse
import enum
import multiprocessing as mp

import torch

from vizbert.data import DATA_WORKSPACE_CLASSES


__all__ = ['ArgumentParserBuilder', 'opt', 'OptionEnum']


def _make_parser_setter(option, key):
    def fn(value):
        option.kwargs[key] = value
        return option
    return fn


class ArgumentParserOption(object):

    def __init__(self, *args, default_init=None, choices_init=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if default_init is not None:
            self.kwargs.setdefault('default', default_init())
        if choices_init is not None:
            self.kwargs.setdefault('choices', choices_init())


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


def init_dataset_choices():
    return list(DATA_WORKSPACE_CLASSES.keys())


class OptionEnum(enum.Enum):
    DATA_FOLDER = opt('--data-folder', '-df', type=Path, required=True)
    OUTPUT_FOLDER = opt('--output-folder', '-of', type=Path, required=True)
    OUTPUT_FILE = opt('--output-file', '-o', type=Path, required=True)
    WORKSPACE = opt('--workspace', '-w', type=Path, required=True)
    LAYER_IDX = opt('--layer-idx', '-l', type=int, required=True)
    PROBE_RANK = opt('--probe-rank', type=int, default=2)
    MODEL = opt('--model', default='bert-base-cased', type=str)
    DEVICE = opt('--device', type=torch.device, default='cuda:0')
    LR = opt('--lr', type=float, default=5e-4)
    NUM_WORKERS = opt('--num-workers', type=int, default_init=mp.cpu_count)
    NUM_EPOCHS = opt('--num-epochs', type=int, default=1)
    BATCH_SIZE = opt('--batch-size', '-bsz', type=int, default=16)
    LOAD_WEIGHTS = opt('--load-weights', action='store_true')
    EVAL_ONLY = opt('--eval-only', action='store_true')
    USE_ZMT = opt('--use-zmt', action='store_true')
    OPTIMIZE_MEAN = opt('--optimize-mean', action='store_true')
    EVAL_BATCH_SIZE = opt('--eval-batch-size', default=16, type=int)
    DATASET = opt('--dataset', '-d', type=str, choices_init=init_dataset_choices, required=True)
    MAX_SEQ_LEN = opt('--max-seq-len', '-msl', type=int, default=128)
    INVERSE = opt('--inverse', action='store_true')
