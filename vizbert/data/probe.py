from torch.nn.init import xavier_uniform_
import torch
import torch.nn as nn


class InnerProductProbe(nn.Module):

    def __init__(self, length: int, max_rank: int = None):
        super().__init__()
        self.length = length
        if max_rank is None:
            max_rank = length
        self.b = nn.Parameter(torch.Tensor(max_rank, length), requires_grad=True)
        xavier_uniform_(self.b.data)

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
        return (mask * torch.abs(scores - labels)).sum() / mask.sum()
