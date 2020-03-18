import torch
import torch.nn as nn

from vizbert.utils import orth_tensor, full_batch_gs


__all__ = ['InnerProductProbe', 'DistanceMatrixLoss', 'ProjectionPursuitProbe']


class InnerProductProbe(nn.Module):

    def __init__(self, length: int, max_rank: int = None):
        super().__init__()
        self.length = length
        if max_rank is None:
            max_rank = length
        self.b = nn.Parameter(torch.Tensor(max_rank, length).uniform_(-0.05, 0.05), requires_grad=True)

    def forward(self, x):
        seq_len = x.size(1)
        x = torch.einsum('gh,bih->big', self.b, x)
        x = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        y = x.clone().permute(0, 2, 1, 3)
        z = x - y
        return torch.einsum('bijg,bijg->bij', z, z)


class DistanceMatrixLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores, labels, mask):
        sq_lengths = mask.view(mask.size(0), -1).sum(1)
        l1_diff = (mask * torch.abs(scores - labels)).view(labels.size(0), -1).sum(1)
        return torch.mean(l1_diff / sq_lengths)


class ProjectionPursuitProbe(nn.Module):

    def __init__(self, num_features, rank=None, normalize=False):
        super().__init__()
        self.num_features = num_features
        if rank is None:
            rank = self.num_features
        self.rank = rank
        self.normalize = normalize
        self.probe = nn.Parameter(torch.Tensor(num_features, rank).uniform_(-0.05, 0.05), requires_grad=True)

    def orth_probe(self):
        return orth_tensor(self.probe)

    def forward(self, hidden_states: torch.Tensor):
        if self.normalize:
            old_norms = hidden_states.norm(dim=2).unsqueeze(-1)
        hidden_states = full_batch_gs(self.orth_probe(), hidden_states)
        if self.normalize:
            norms = hidden_states.norm(dim=2).unsqueeze(-1)
            hidden_states = (hidden_states / norms) * old_norms
        return hidden_states
