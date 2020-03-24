import numpy as np
import torch
import torch.nn as nn


__all__ = ['ZeroMeanTransform']


class ZeroMeanTransform(nn.Module):

    def __init__(self, num_features: int, dim=0):
        super().__init__()
        self.register_buffer('total', torch.zeros(1))
        self.register_buffer('mean', torch.zeros(num_features))
        self.dim = dim
        self.num_features = num_features

    def update(self, data, mask=None):
        with torch.no_grad():
            if mask is not None:
                data = data * mask
                mask_size = mask.sum().item() // self.num_features
            else:
                mask_size = data.numel() // self.num_features

            new_sizes = []
            new_idx = 0
            if self.dim != 0:
                new_sizes.append(-1)
                new_idx = 1
            new_sizes.append(self.num_features)
            if self.dim < data.dim() - 1:
                new_sizes.append(np.prod(data.size()[self.dim + 1:]))
            data = data.view(*new_sizes).contiguous()
            if data.dim() == 3:
                data = data.permute(1, 0, 2).contiguous()
            elif data.dim() == 2 and new_idx == 1:
                data = data.permute(1, 0).contiguous()
            elif data.dim() == 1:
                data = data.unsqueeze(-1)
            self.mean = (data.sum(1) + self.mean * self.total) / (self.total + mask_size)
            self.total += mask_size

    def forward(self, x, shift='subtract'):
        mean = self.mean
        for _ in range(self.dim):
            mean = mean.unsqueeze(0)
        for _ in range(self.dim, x.dim()):
            mean = mean.unsqueeze(-1)
        if shift == 'subtract':
            return x - self.mean
        else:
            return x + self.mean
