from functools import partial

from torch.distributions import Categorical
import torch
import torch.nn.functional as F
import torch.nn as nn


__all__ = ['DistanceMatrixLoss', 'EntropyLoss', 'ReconstructionLoss', 'MaskedConceptLoss', 'ClassificationLoss']


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


class ReconstructionLoss(nn.Module):

    def __init__(self, multilabel=False, regression=False):
        super().__init__()
        self.criterion = nn.MSELoss() if regression else nn.L1Loss()
        self.multilabel = multilabel
        self.regression = regression

    def forward(self, scores, gold_scores, *args):
        if self.regression:
            return self.criterion(F.sigmoid(scores), F.sigmoid(gold_scores))
        else:
            return self.criterion(F.softmax(scores, -1), F.softmax(gold_scores, -1))


class ClassificationLoss(nn.Module):

    def __init__(self, multilabel=False):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    def forward(self, scores, labels, *args):
        return self.criterion(scores, labels)


class MaskedConceptLoss(nn.Module):

    def __init__(self, multilabel=False, weight=1, inverse=False):
        super().__init__()
        self.criterion = nn.L1Loss()  # TVD
        self.multilabel = multilabel
        self.weight = weight
        self.inverse = inverse

    def _multilabel_op(self, x, gold=False):
        x = x.sigmoid()
        y = 1 - x
        if not gold:
            x = x.log()
            y = y.log()
        return torch.stack((x, y), -1)

    def forward(self, scores: torch.Tensor, gold_scores: torch.Tensor, mask: torch.Tensor):
        gold_scores[:, mask] = -1000
        if mask is not None:
            scores[:, mask] *= self.weight
        if self.multilabel:
            scores = scores.sigmoid()
            gold_scores = gold_scores.sigmoid()
        else:
            scores = scores.softmax(-1)
            gold_scores = gold_scores.softmax(-1)
            # if self.inverse and mask is not None:
            #     mask_labels = set(mask.tolist())
            #     inv_mask = list(filter(lambda x: x not in mask_labels, range(scores.size(1))))
            #     inv_mask = torch.tensor(inv_mask).to(mask.device)
            #     # masked_scores = scores[:, mask].sum(-1).unsqueeze(-1)
            #     # masked_gold_scores = gold_scores[:, mask].sum(-1).unsqueeze(-1)
            #     # o_scores = scores[:, inv_mask]
            #     # o_gold_scores = gold_scores[:, inv_mask]
            #     # scores = torch.cat((o_scores, masked_scores), 1)
            #     # gold_scores = torch.cat((o_gold_scores, masked_gold_scores), 1)
            #     gold_scores[:, inv_mask] = ((1 - gold_scores[:, mask].sum(-1)) / len(inv_mask)).unsqueeze(-1)
        tvd = 0.5 * self.criterion(scores, gold_scores)
        return tvd
