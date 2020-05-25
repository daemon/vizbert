from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Sequence
import re

from transformers import PreTrainedTokenizer
import torch.utils.data as tud
import torch
import pandas as pd

from .metric import MetricDataset


__all__ = ['DataFrameDataset',
           'Sst2Workspace',
           'ClassificationCollator',
           'ReutersWorkspace',
           'LabeledTextBatch',
           'ImdbWorkspace',
           'AapdWorkspace',
           'ColaWorkspace',
           'Sst5Workspace',
           'MismatchedMnliWorkspace',
           'MatchedMnliWorkspace',
           'CleanedImdbDytangWorkspace',
           'ImdbDytangWorkspace',
           'CleanedImdbWorkspace',
           'SENTENCE_SEPARATOR']

INDEX_COLUMN = 'index'
SENTENCE1_COLUMN = 'sentence1'
SENTENCE2_COLUMN = 'sentence2'
LABEL_COLUMN = 'label'
SCORE_COLUMN = 'score'
LABEL_NAMES_COLUMN = 'label_names'
SENTENCE_SEPARATOR = '<sssss>'


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

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 multilabel=False,
                 max_length=128,
                 presplit=False):
        self.tokenizer = tokenizer
        if presplit:
            self.tokenizer.do_basic_tokenize = False
        self.multilabel = multilabel
        self.max_length = max_length
        self.presplit = presplit

    def __call__(self, examples: Sequence[pd.DataFrame]):
        token_ids = []
        masks = []
        raw_text = []
        labels = []
        segment_ids = []
        for ex in examples:
            dd = self.tokenizer.encode_plus(ex[SENTENCE1_COLUMN].replace(f' {SENTENCE_SEPARATOR}', ''),
                                            text_pair=ex[SENTENCE2_COLUMN].replace(f' {SENTENCE_SEPARATOR}', '') \
                                                if SENTENCE2_COLUMN in ex else None,
                                            max_length=self.max_length)
            enc = dd['input_ids']
            tt_ids = dd['token_type_ids']
            token_ids.append(enc)
            masks.append([1] * len(enc))
            raw_text.append(ex[SENTENCE1_COLUMN])
            if SENTENCE2_COLUMN in ex:
                raw_text[-1] = f'{raw_text[-1]}\t{ex[SENTENCE2_COLUMN]}'
            segment_ids.append(tt_ids)
            if LABEL_COLUMN in ex:
                if self.multilabel:
                    labels.append(list(map(int, ex[LABEL_COLUMN])))
                else:
                    labels.append(ex[LABEL_COLUMN])
            elif SCORE_COLUMN in ex:
                labels.append(float(ex[SCORE_COLUMN]))
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


@dataclass
class StsbWorkspace(object):
    folder: Path

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename, set_type):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False)
            if set_type in {'dev', 'train'}:
                df.columns = ['uu1', 'uu2', 'uu3', 'uu4', 'uu5', 'uu6', 'uu7', SENTENCE1_COLUMN, SENTENCE2_COLUMN, SCORE_COLUMN]
                labeled = True
            else:
                df.columns = ['uu1', 'uu2', 'uu3', 'uu4', 'uu5', 'uu6', 'uu7', SENTENCE1_COLUMN, SENTENCE2_COLUMN]
                labeled = False
            return DataFrameDataset(df, num_labels=1, labeled=labeled, metrics=('pearsonr', 'spearmanr'))
        return [load(str(self.folder / f'{set_type}.tsv'), set_type) for set_type in splits]


class DataFrameDataset(MetricDataset, tud.Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 num_labels: int,
                 labeled: bool = False,
                 multilabel: bool = False,
                 label_map: Sequence[str] = None,
                 metrics: Sequence[str] = None,
                 presplit: bool = False):
        MetricDataset.__init__(self, metrics)
        self.dataframe = dataframe
        self.num_labels = num_labels
        self.labeled = labeled
        self.multilabel = multilabel
        self.l2idx = label_map
        self.idx2l = {v: k for v, k in enumerate(label_map)} if label_map else None
        self.presplit = presplit

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


class ImdbCleanerWorkspace(object):
    double_space = re.compile(r'  +')
    no_sssss = re.compile(r'\s<sssss>\s*$')

    def __init__(self):
        from .pattern import ImdbSpuriousMatcher
        self.matcher = ImdbSpuriousMatcher()

    def clean(self, text: str):
        text = self.matcher.clean(text)
        return self.no_sssss.sub('', self.double_space.sub('', text))


@dataclass
class ImdbWorkspace(ImdbCleanerWorkspace):
    folder: Path
    cleaned: bool = False

    def __post_init__(self):
        super().__init__()

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, header=None, dtype=str)
            df.columns = [LABEL_COLUMN, SENTENCE1_COLUMN]
            df[LABEL_COLUMN] = list(map(lambda x: x.index('1'), df[LABEL_COLUMN]))
            if self.cleaned:
                df[SENTENCE1_COLUMN] = list(map(self.clean, df[SENTENCE1_COLUMN]))
            return DataFrameDataset(df, num_labels=10, labeled=True, multilabel=False, metrics=('accuracy',))
        return [load(str(self.folder / f'{set_type}.tsv')) for set_type in splits]


@dataclass
class ImdbDytangWorkspace(ImdbCleanerWorkspace):
    folder: Path
    cleaned: bool = False

    def __post_init__(self):
        super().__init__()

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, header=None, dtype=str)
            df.columns = ['uu1', 'uu2', 'uu3', 'uu4', LABEL_COLUMN, 'uu5', SENTENCE1_COLUMN]
            df[LABEL_COLUMN] = list(map(lambda x: int(x) - 1, df[LABEL_COLUMN]))
            if self.cleaned:
                df[SENTENCE1_COLUMN] = list(map(self.clean, df[SENTENCE1_COLUMN]))
            return DataFrameDataset(df, num_labels=10, labeled=True, multilabel=False, presplit=True, metrics=('accuracy',))
        return [load(str(self.folder / f'imdb.{set_type}.txt.ss')) for set_type in splits]


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


@dataclass
class MnliWorkspace(object):
    folder: Path
    matched: bool = False

    def load_splits(self, splits=('train', 'dev', 'test')):
        def load(filename, set_type):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False, dtype=str, na_filter=False)
            cols = [INDEX_COLUMN, 'uu1', 'uu2', 'uu3', 'uu4', 'uu5', 'uu6', 'uu7', SENTENCE1_COLUMN,
                    SENTENCE2_COLUMN]
            if 'train' in set_type:
                cols.extend(['uu8', LABEL_COLUMN])
            elif 'dev' in set_type:
                cols.extend(['uu8', 'uu9', 'uu10', 'uu11', 'uu12', LABEL_COLUMN])
            df.columns = cols
            if 'test' not in set_type:
                df[LABEL_COLUMN] = list(map(label_map.__getitem__, df[LABEL_COLUMN]))
            return DataFrameDataset(df, num_labels=3, labeled='test' not in set_type, multilabel=False, metrics=('accuracy',))

        label_map = dict(contradiction=0, neutral=2, entailment=1)
        splits = [s if s == 'train' else f'{s}_{"" if self.matched else "mis"}matched' for s in splits]
        return [load(str(self.folder / f'{set_type}.tsv'), set_type) for set_type in splits]


MatchedMnliWorkspace = partial(MnliWorkspace, matched=True)
MismatchedMnliWorkspace = partial(MnliWorkspace, matched=False)
CleanedImdbDytangWorkspace = partial(ImdbDytangWorkspace, cleaned=True)
CleanedImdbWorkspace = partial(ImdbWorkspace, cleaned=True)
