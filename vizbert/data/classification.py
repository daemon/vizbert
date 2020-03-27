from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch.utils.data as tud
import torch
import pandas as pd

from .metric import MetricDataset, METRIC_MAP, Metric


__all__ = ['DataFrameDataset',
           'Sst2Workspace',
           'ClassificationCollator',
           'ReutersWorkspace',
           'LabeledTextBatch',
           'ImdbWorkspace',
           'AapdWorkspace',
           'ColaWorkspace',
           'Sst5Workspace']

INDEX_COLUMN = 'index'
SENTENCE1_COLUMN = 'sentence1'
SENTENCE2_COLUMN = 'sentence2'
LABEL_COLUMN = 'label'
LABEL_NAMES_COLUMN = 'label_names'


@dataclass
class LabeledTextBatch(object):
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    raw_text: Sequence[str]
    segment_ids: torch.Tensor
    labels: torch.Tensor = None
    multilabel: bool = False

    def pin_memory(self):
        self.token_ids.pin_memory()
        self.segment_ids.pin_memory()
        if self.labels is not None:
            self.labels.pin_memory()
        self.attention_mask.pin_memory()
        return self


class ClassificationCollator(object):

    def __init__(self, tokenizer, multilabel=False, max_length=128):
        self.tokenizer = tokenizer
        self.multilabel = multilabel
        self.max_length = max_length

    def __call__(self, examples: Sequence[pd.DataFrame]):
        token_ids = []
        masks = []
        raw_text = []
        labels = []
        segment_ids = []
        for ex in examples:
            dd = self.tokenizer.encode_plus(ex[SENTENCE1_COLUMN],
                                            text_pair=ex[SENTENCE2_COLUMN] if SENTENCE2_COLUMN in ex else None,
                                            max_length=self.max_length)
            enc = dd['input_ids']
            tt_ids = dd['token_type_ids']
            token_ids.append(enc)
            masks.append([1] * len(enc))
            raw_text.append(ex[SENTENCE1_COLUMN])
            segment_ids.append(tt_ids)
            if LABEL_COLUMN in ex:
                if self.multilabel:
                    labels.append(list(map(int, ex[LABEL_COLUMN])))
                else:
                    labels.append(ex[LABEL_COLUMN])
        max_len = max(len(x) for x in token_ids)
        masks = torch.tensor([x + ([0] * (max_len - len(x))) for x in masks])
        segment_ids = torch.tensor([x + ([0] * (max_len - len(x))) for x in segment_ids])
        token_ids = torch.tensor([x + ([0] * (max_len - len(x))) for x in token_ids])
        return LabeledTextBatch(token_ids, masks, raw_text, segment_ids, labels=torch.tensor(labels) if labels else None, multilabel=self.multilabel)


@dataclass
class ColaWorkspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename, set_type):
            use_header = dict(header=None) if set_type in {'dev', 'train'} else {}
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, **use_header)
            if set_type in {'dev', 'train'}:
                df.columns = ['unused1', LABEL_COLUMN, 'unused2', SENTENCE1_COLUMN]
                labeled = True
            else:
                df.columns = [INDEX_COLUMN, SENTENCE1_COLUMN]
                labeled = False
            return DataFrameDataset(df, num_labels=2, labeled=labeled, metrics=('mcc', 'accuracy'))
        return [load(str(self.folder / f'{set_type}.tsv'), set_type) for set_type in splits]


@dataclass
class Sst2Workspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename, set_type):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False)
            if set_type in {'dev', 'train'}:
                df.columns = [SENTENCE1_COLUMN, LABEL_COLUMN]
                labeled = True
            else:
                df.columns = [INDEX_COLUMN, SENTENCE1_COLUMN]
                labeled = False
            return DataFrameDataset(df, num_labels=2, labeled=labeled, metrics=('accuracy',))
        return [load(str(self.folder / f'{set_type}.tsv'), set_type) for set_type in splits]


class DataFrameDataset(tud.Dataset, MetricDataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 num_labels: int,
                 labeled: bool = False,
                 multilabel: bool = False,
                 label_map: Sequence[str] = None,
                 metrics: Sequence[str] = None):
        self.dataframe = dataframe
        self.num_labels = num_labels
        self.labeled = labeled
        self.multilabel = multilabel
        self.l2idx = label_map
        self.idx2l = {v: k for v, k in enumerate(label_map)} if label_map else None
        if metrics is None:
            metrics = []
        self._metrics = [METRIC_MAP[x]() for x in metrics]

    @property
    def metrics(self) -> Sequence[Metric]:
        return self._metrics

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.loc[idx]


@dataclass
class ReutersWorkspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, header=None, dtype=str)
            df.columns = [LABEL_COLUMN, SENTENCE1_COLUMN]
            return DataFrameDataset(df, num_labels=90, labeled=True, multilabel=True, metrics=('recall', 'precision'))
        return [load(str(self.folder / f'{set_type}.tsv')) for set_type in splits]


@dataclass
class AapdWorkspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, dtype=str)
            df.columns = [SENTENCE1_COLUMN, LABEL_COLUMN, LABEL_NAMES_COLUMN]
            return DataFrameDataset(df, num_labels=54, labeled=True, multilabel=True, label_map=idx2l, metrics=('recall', 'precision'))
        idx2l = []
        with open(str(self.folder / 'labels.csv')) as f:
            for idx, line in enumerate(f.readlines()):
                label = line.split(',')[1].strip()
                idx2l.append(label)
        return [load(str(self.folder / f'{set_type}.tsv')) for set_type in splits]


@dataclass
class ImdbWorkspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, header=None, dtype=str, metrics=('accuracy',))
            df.columns = [LABEL_COLUMN, SENTENCE1_COLUMN]
            df[LABEL_COLUMN] = list(map(lambda x: x.index('1'), df[LABEL_COLUMN]))
            return DataFrameDataset(df, num_labels=10, labeled=True, multilabel=False)
        return [load(str(self.folder / f'{set_type}.tsv')) for set_type in splits]


@dataclass
class Sst5Workspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, header=None, dtype=str)
            df.columns = [LABEL_COLUMN, SENTENCE1_COLUMN]
            df[LABEL_COLUMN] = list(map(int, df[LABEL_COLUMN]))
            return DataFrameDataset(df, num_labels=5, labeled=True, multilabel=False, metrics=('accuracy',))
        return [load(str(self.folder / f'stsa.fine.{set_type}')) for set_type in splits]
