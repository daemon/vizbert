from torch.distributions import Categorical
import torch
import torch.nn as nn


__all__ = ['DistanceMatrixLoss', 'EntropyLoss', 'ClassificationLoss']


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
        return self.criterion(scores, labels)


class MaskedConceptLoss(nn.Module):

    def __init__(self, multilabel=False):
        super().__init__()
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.multilabel = multilabel

    def forward(self, scores: torch.Tensor, gold_scores: torch.Tensor, mask: torch.Tensor):
        if self.multilabel:
            recon_scores = scores[1 - mask].unsqueeze(-1).sigmoid().log()
            recon_gold_scores = gold_scores[1 - mask].unsqueeze(-1).sigmoid()
            destruct_scores = scores[mask].unsqueeze(-1).sigmoid().log()
        else:
            recon_scores = scores[1 - mask].log_softmax(-1)
            recon_gold_scores = gold_scores[1 - mask].softmax(-1)
            destruct_scores = scores[mask].unsqueeze(-1).log_softmax(-1)
        recon_loss = self.kldiv_loss(recon_scores, recon_gold_scores)
        destruct_loss = self.kldiv_loss(destruct_scores, torch.zeros_like(scores[mask]).to(scores.device))
        return recon_loss + destruct_loss
