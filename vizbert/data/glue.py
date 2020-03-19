from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch.utils.data as tud
import torch
import pandas as pd


__all__ = ['DataFrameDataset', 'GlueWorkspace', 'GlueCollator']
INDEX_COLUMN = 'index'
SENTENCE1_COLUMN = 'sentence1'
SENTENCE2_COLUMN = 'sentence2'
LABEL_COLUMN = 'label'


@dataclass
class LabeledTextBatch(object):
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    raw_text: Sequence[str]
    segment_ids: torch.Tensor
    labels: torch.Tensor = None

    def pin_memory(self):
        self.token_ids.pin_memory()
        if self.labels is not None:
            self.labels.pin_memory()
        self.attention_mask.pin_memory()
        return self


class GlueCollator(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: Sequence[pd.DataFrame]):
        token_ids = []
        masks = []
        raw_text = []
        labels = []
        segment_ids = []
        for ex in examples:
            dd = self.tokenizer.encode_plus(ex[SENTENCE1_COLUMN], text_pair=ex[SENTENCE2_COLUMN] if SENTENCE2_COLUMN in ex else None)
            enc = dd['input_ids']
            tt_ids = dd['token_type_ids']
            token_ids.append(enc)
            masks.append([1] * len(enc))
            raw_text.append(ex[SENTENCE1_COLUMN])
            segment_ids.append(tt_ids)
            if LABEL_COLUMN in ex:
                labels.append(ex[LABEL_COLUMN])
        max_len = max(len(x) for x in token_ids)
        masks = [x.extend([0] * (max_len - len(x))) for x in masks]
        segment_ids = [x.extend([0] * (max_len - len(x))) for x in segment_ids]
        token_ids = [x.extend([0] * (max_len - len(x))) for x in token_ids]
        return LabeledTextBatch(torch.tensor(token_ids),
                                torch.tensor(masks),
                                raw_text,
                                torch.tensor(segment_ids),
                                labels=torch.tensor(labels) if labels else None)


class DataFrameDataset(tud.Dataset):

    def __init__(self, dataframe: pd.DataFrame, num_labels: int, labeled: bool = False):
        self.dataframe = dataframe
        self.num_labels = num_labels
        self.labeled = labeled

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe[idx]


@dataclass
class GlueWorkspace(object):
    folder: Path

    def load_sst2_splits(self, splits=('train', 'dev', 'test')):
        def load(filename, set_type):
            df = pd.read_csv(filename, sep='\t', quoting=3, error_bad_lines=False)
            if set_type in {'dev', 'train'}:
                df.columns = [SENTENCE1_COLUMN, LABEL_COLUMN]
                labeled = True
            else:
                df.columns = [INDEX_COLUMN, SENTENCE1_COLUMN]
                labeled = False
            return DataFrameDataset(df, num_labels=2, labeled=labeled)
        folder = self.folder / 'SST-2'
        return [load(str(folder / f'{set_type}.tsv'), set_type) for set_type in splits]
