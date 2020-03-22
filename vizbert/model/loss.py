from functools import partial

from torch.distributions import Categorical
import torch
import torch.nn as nn


__all__ = ['DistanceMatrixLoss', 'EntropyLoss', 'ClassificationLoss', 'MaskedConceptLoss']


class EntropyLoss(nn.Module):

    def __init__(self, mode='max', reduction='mean'):
        super().__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, *args, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(scores).to(scores.device)
        if scores.dim() > 1:
            attention_mask = (1 - attention_mask) * -10000
            scores = scores + attention_mask
        entropy = Categorical(logits=scores).entropy()
        if self.mode == 'max':
            entropy = -entropy
        if self.reduction == 'mean':
            while entropy.dim() > 1:
                entropy = entropy.mean(entropy.dim() - 2)
            entropy = entropy.mean()
        return entropy


class DistanceMatrixLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, scores, labels, mask):
        sq_lengths = mask.view(mask.size(0), -1).sum(1)
        l1_diff = (mask * torch.abs(scores - labels)).view(labels.size(0), -1).sum(1)
        return torch.mean(l1_diff / sq_lengths)


class ClassificationLoss(nn.Module):

    def __init__(self, multilabel=False):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
        self.multilabel = multilabel

    def forward(self, scores, labels):
        if self.multilabel:
            labels = labels.float()
        return self.criterion(scores, labels)


class MaskedConceptLoss(nn.Module):

    def __init__(self, multilabel=False):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.multilabel = multilabel

    def _multilabel_op(self, x, gold=False):
        x = x.sigmoid()
        y = 1 - x
        if gold:
            x = x.log()
            y = y.log()
        return torch.stack((x, y), -1)

    def forward(self, scores: torch.Tensor, gold_scores: torch.Tensor, mask: torch.Tensor):
        gold_scores[mask] = -1000
        if self.multilabel:
            scores = scores.sigmoid()
            gold_scores = gold_scores.sigmoid()
        else:
            scores = scores.log_softmax(-1)
            gold_scores = gold_scores.log_softmax(-1)
        return self.criterion(scores, gold_scores)
