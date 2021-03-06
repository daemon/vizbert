from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil

from torch.utils.tensorboard import SummaryWriter
import torch

from .classification import Sst2Workspace, ReutersWorkspace, ImdbWorkspace, AapdWorkspace, Sst5Workspace, ColaWorkspace,\
    MismatchedMnliWorkspace, MatchedMnliWorkspace, ImdbDytangWorkspace, CleanedImdbDytangWorkspace, CleanedImdbWorkspace, StsbWorkspace
from vizbert.data import ConllDataset
from vizbert.utils import JSON_PRIMITIVES


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
    cli_name = 'cli.json'

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

    def write_args(self, args: argparse.Namespace):
        cli_dict = {k: repr(v) if type(v)
                                  not in JSON_PRIMITIVES else v for k, v in vars(args).items()}
        with open(str(self.folder / self.cli_name), 'w') as f:
            json.dump(cli_dict, f, indent=4)

    def save_model(self, model: torch.nn.Module, model_name=None):
        if model_name is None:
            model_name = Path(self.model_name)
        else:
            model_name = Path(model_name).with_suffix('.pt')
        torch.save(model.state_dict(), self.folder / model_name.name)

    def load_model(self, model: torch.nn.Module, model_name=None):
        if model_name is None:
            model_name = Path(self.model_name)
        else:
            model_name = Path(model_name).with_suffix('.pt')
        state_dict = torch.load(self.folder / model_name.name, lambda s, l: s)
        model.load_state_dict(state_dict, strict=False)
        return state_dict


DATA_WORKSPACE_CLASSES = dict(conll=ConllWorkspace,
                              reuters=ReutersWorkspace,
                              sst2=Sst2Workspace,
                              imdb=ImdbWorkspace,
                              aapd=AapdWorkspace,
                              sst5=Sst5Workspace,
                              cola=ColaWorkspace,
                              mnli_mm=MismatchedMnliWorkspace,
                              mnli_m=MatchedMnliWorkspace,
                              imdb_dyt=ImdbDytangWorkspace,
                              imdb_dyt_clean=CleanedImdbDytangWorkspace,
                              imdb_clean=CleanedImdbWorkspace,
                              stsb=StsbWorkspace)
