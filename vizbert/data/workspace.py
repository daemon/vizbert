from dataclasses import dataclass
from pathlib import Path

import torch

from vizbert.data import ConllDataset


@dataclass
class DataWorkspace(object):
    folder: Path
    train_name = 'train.gold.conll'
    dev_name = 'dev.gold.conll'
    test_name = 'test.gold.conll'

    def make_hidden_state_filename(self, filename: str, layer_idx):
        return (self.folder / filename).with_suffix(f'.l{layer_idx}.pt')

    def load_attached_hidden_state_dataset(self, filename: str, layer_idx):
        ds = ConllDataset.from_file(self.folder / filename)
        hidden_states = torch.load(self.make_hidden_state_filename(filename, layer_idx))['hidden_state']
        hid_joined = []
        for hid in hidden_states:
            x = [y.squeeze(0) for y in torch.split(hid, 1)]
            if hid.dim() > 3:
                x = [y.permute(1, 0, 2).contiguous().view(y.size(2), -1) for y in x]
            hid_joined.extend(x)
        ds.attach('hidden_state', hid_joined)
        return ds

    def load_conll_splits(self, attach_hidden=False, layer_idx=None):
        if not attach_hidden:
            return ConllDataset.from_file(self.folder / self.train_name),\
                   ConllDataset.from_file(self.folder / self.dev_name),\
                   ConllDataset.from_file(self.folder / self.test_name)
        return self.load_attached_hidden_state_dataset(self.train_name, layer_idx),\
               self.load_attached_hidden_state_dataset(self.dev_name, layer_idx),\
               self.load_attached_hidden_state_dataset(self.test_name, layer_idx)
