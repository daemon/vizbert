import torch
import torch.nn as nn


class InnerProductProbe(nn.Module):

    def __init__(self, length: int):
        super().__init__()
        self.length = length
        self.b = nn.Parameter(torch.Tensor(length, length).uniform_(-0.1, 0.1), requires_grad=True)

    def forward(self, x):
        seq_len = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        y = x.clone().permute(0, 2, 1, 3)
        z = x - y
        h = torch.einsum('gh,bijh->bijg', self.b, z)
        distance_matrix = torch.einsum('bijg,bijg->bij', h, h)
        return distance_matrix


class DistanceMatrixLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores, labels, mask):
        return (mask * ((scores - labels) ** 2)).sum() / mask.sum()
