import math

import torch
import torch.nn as nn

from vizbert.data import ZeroMeanTransform
from vizbert.utils import orth_tensor, full_batch_gs, full_batch_proj


__all__ = ['InnerProductProbe', 'ProjectionPursuitProbe', 'LowRankProjectionTransform']


class InnerProductProbe(nn.Module):

    def __init__(self, length: int, max_rank: int = None):
        super().__init__()
        self.length = length
        if max_rank is None:
            max_rank = length
        self.b = nn.Parameter(torch.empty(max_rank, length, dtype=torch.float32).uniform_(-0.05, 0.05), requires_grad=True)

    def forward(self, x):
        seq_len = x.size(1)
        x = torch.einsum('gh,bih->big', self.b, x)
        x = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        y = x.clone().permute(0, 2, 1, 3)
        z = x - y
        return torch.einsum('bijg,bijg->bij', z, z)


class LowRankProjectionTransform(nn.Module):

    def __init__(self, num_features: int, rank: int):
        super().__init__()
        self.probe_params_ = [nn.Parameter(torch.empty(num_features, dtype=torch.float32).uniform_(-0.05, 0.05),
                                           requires_grad=True) for _ in range(rank)]
        self.probe_params = nn.ParameterList(self.probe_params_)

    @property
    def orth_probe(self):
        return orth_tensor(torch.stack(self.probe_params_, 1))

    def forward(self, input: torch.Tensor):
        return full_batch_proj(self.orth_probe, input)


class ProjectionPursuitProbe(nn.Module):

    def __init__(self,
                 num_features,
                 rank=None,
                 normalize=False,
                 orthogonalize=True,
                 mask_first=False,
                 zero_mean_transform: ZeroMeanTransform = None,
                 optimize_mean: bool = False,
                 inverse: bool = False):
        super().__init__()
        self.num_features = num_features
        if rank is None:
            rank = self.num_features
        self.zmt = zero_mean_transform
        self.rank = rank
        self.normalize = normalize
        self.mask_first = mask_first
        self.optimize_mean = optimize_mean
        if optimize_mean:
            self.mean = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.probe_ = [nn.Parameter(torch.empty(num_features, dtype=torch.float32).uniform_(-0.05, 0.05), requires_grad=True) for _ in range(rank)]
        self.probe_params = nn.ParameterList(self.probe_)
        self.orthogonalize = orthogonalize
        self.inverse = inverse

    def l1_penalty(self, weight):
        return weight * self.probe_params[0].norm(p=1)

    @property
    def orth_probe(self):
        return orth_tensor(torch.stack(self.probe_, 1))

    @property
    def probe(self):
        return torch.stack(self.probe_, 1)

    def forward(self, hidden_states: torch.Tensor):
        if self.optimize_mean:
            hidden_states = hidden_states - self.mean.unsqueeze(0).unsqueeze(0)
        if self.zmt is not None:
            hidden_states = self.zmt(hidden_states, shift='subtract')
        if self.normalize:
            old_norms = hidden_states.norm(dim=2).unsqueeze(-1)
        if self.mask_first:
            first_state = hidden_states[:, 0]
        probe = self.orth_probe if self.orthogonalize else self.probe
        if self.inverse:
            hidden_states = full_batch_proj(probe, hidden_states)
        else:
            hidden_states = full_batch_gs(probe, hidden_states)
        if self.normalize:
            norms = hidden_states.norm(dim=2).unsqueeze(-1)
            hidden_states = (hidden_states / norms) * old_norms
        if self.mask_first:
            hidden_states[:, 0] = first_state
        if self.zmt is not None:
            hidden_states = self.zmt(hidden_states, shift='add')
        if self.optimize_mean:
            hidden_states = hidden_states + self.mean.unsqueeze(0).unsqueeze(0)
        return hidden_states
