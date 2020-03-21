from dataclasses import dataclass
from pathlib import Path
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch

from .classification import Sst2Workspace, ReutersWorkspace
from vizbert.data import ConllDataset


__all__ = ['ConllWorkspace', 'TrainingWorkspace', 'DATA_WORKSPACE_CLASSES']


@dataclass
class ConllWorkspace(object):
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
            hid_joined.extend(y.squeeze(0) for y in torch.split(hid, 1))
        ds.attach('hidden_state', hid_joined)
        return ds

    def load_splits(self, attach_hidden=False, layer_idx=None, splits=None):
        if splits is None:
            splits = (self.train_name, self.dev_name, self.test_name)
        if not attach_hidden:
            return [ConllDataset.from_file(self.folder / x) for x in splits]
        return [self.load_attached_hidden_state_dataset(x, layer_idx) for x in splits]


@dataclass
class TrainingWorkspace(object):
    folder: Path
    model_name = 'model.pt'

    @property
    def model_path(self):
        return self.folder / self.model_name

    def __post_init__(self):
        self.log_path = self.folder / 'logs'
        try:
            self.folder.mkdir()
        except:
            pass
        try:
            shutil.rmtree(str(self.log_path), ignore_errors=True)
        except:
            pass
        self.summary_writer = SummaryWriter(str(self.log_path))

    def save_model(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.folder / self.model_name)

    def load_model(self, model: torch.nn.Module):
        model.load_state_dict(torch.load(self.folder / self.model_name, lambda s, l: s))


DATA_WORKSPACE_CLASSES = dict(conll=ConllWorkspace, reuters=ReutersWorkspace, sst2=Sst2Workspace)
