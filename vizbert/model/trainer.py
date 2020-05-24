from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Tuple, Callable, Dict

from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.utils.data as tud

from vizbert.data import TrainingWorkspace


__all__ = ['LOSS_KEY', 'ModelTrainer', 'LOSS_SIZE_KEY', 'InputLossFeeder']
LOSS_KEY = 'loss'
LOSS_SIZE_KEY = 'loss_size'


InputLossFeeder = Callable[['ModelTrainer', Any], Dict]


@dataclass
class ModelTrainer(object):
    loaders: Tuple[tud.DataLoader, tud.DataLoader, tud.DataLoader]
    model: nn.Module
    workspace: TrainingWorkspace
    optimizer: Any
    num_epochs: int
    train_feed_loss_callback: InputLossFeeder
    dev_feed_loss_callback: InputLossFeeder = None
    test_feed_loss_callback: InputLossFeeder = None
    scheduler: Any = None
    optimization_limit: int = None

    def __post_init__(self):
        if self.dev_feed_loss_callback is None:
            self.dev_feed_loss_callback = self.train_feed_loss_callback
        if self.test_feed_loss_callback is None:
            self.test_feed_loss_callback = self.train_feed_loss_callback
        self.training = True
        self.step_no = 0
        self.callbacks = []

    @property
    def train_loader(self):
        return self.loaders[0]

    def post_callbacks(self, *callbacks):
        self.callbacks.extend(callbacks)

    @property
    def dev_loader(self):
        return self.loaders[1]

    @property
    def test_loader(self):
        return self.loaders[2]

    def evaluate(self, loader, header='Evaluate'):
        self.training = False
        all_losses = defaultdict(float)
        tot_loss = 0
        tot_len = 0
        self.model.eval()
        cb = self.dev_feed_loss_callback if loader == self.dev_loader else self.test_feed_loss_callback
        with torch.no_grad():
            pbar = tqdm(loader, total=len(loader), position=1)
            for batch in pbar:
                ret = cb(self, batch)
                loss = ret[LOSS_KEY]
                loss_size = ret[LOSS_SIZE_KEY]
                tot_loss += loss.item() * ret[LOSS_SIZE_KEY]
                tot_len += loss_size
                del ret[LOSS_SIZE_KEY]
                curr_mean = tot_loss / tot_len if tot_len > 0 else 0.0
                pbar.set_postfix(dict(loss=f'{curr_mean:.3}'))
                for k, v in ret.items():
                    all_losses[k] += v.item() * loss_size
        all_losses = {k: v / tot_len for k, v in all_losses.items()}
        for loss_name, value in all_losses.items():
            tqdm.write(f'{header},{loss_name.capitalize()},{value:.5}')
        return all_losses

    def train(self, test=True):
        do_exit = False
        for epoch_idx in trange(self.num_epochs, position=0):
            self.training = True
            self.model.train()
            tot_steps = len(self.train_loader)
            if self.optimization_limit is not None:
                tot_steps = min(tot_steps, self.optimization_limit - self.step_no)
            pbar = tqdm(self.train_loader, total=tot_steps, position=1)
            for train_idx, batch in enumerate(pbar):
                loss = self.train_feed_loss_callback(self, batch)[LOSS_KEY]
                self.model.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.step_no += 1
                pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
                for cb in self.callbacks:
                    cb()
                self.callbacks.clear()
                if self.step_no == self.optimization_limit:
                    do_exit = True
                if do_exit:
                    break
            dev_losses = self.evaluate(self.dev_loader, header=epoch_idx + 1)
            if self.scheduler is not None:
                self.scheduler.step(dev_losses[LOSS_KEY])
            for loss_name, value in dev_losses.items():
                self.workspace.summary_writer.add_scalar(f'Dev/{loss_name.capitalize()}', value, epoch_idx)
            self.workspace.save_model(self.model)
            if do_exit:
                break
        if test:
            test_losses = self.evaluate(self.test_loader, header='Test')
            for loss_name, value in test_losses.items():
                self.workspace.summary_writer.add_scalar(f'Test/{loss_name.capitalize()}', value)
