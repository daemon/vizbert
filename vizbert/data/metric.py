from typing import Sequence

from scipy.stats import pearsonr, spearmanr
import torch


METRIC_MAP = {}


class MetricDataset(object):

    def __init__(self, metrics=None):
        if metrics is None:
            metrics = []
        self._metrics = [METRIC_MAP[x]() for x in metrics]

    @property
    def metrics(self) -> Sequence['Metric']:
        return self._metrics

    def evaluate_metrics(self, scores, labels):
        return {x.name: x.evaluate(scores, labels) for x in self.metrics}


class Metric(object):

    def __init_subclass__(cls, **kwargs):
        METRIC_MAP[kwargs['name']] = cls
        cls.name = kwargs['name']

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        raise NotImplementedError

    def evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        if scores.numel() == 0:
            return torch.zeros(1).to(scores.device)
        return self._evaluate(scores, gold)


class MccMetric(Metric, name='mcc'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        tp = ((scores.max(1)[1] == 1) & (gold == 1)).float().sum()
        tn = ((scores.max(1)[1] == 0) & (gold == 0)).float().sum()
        fp = ((scores.max(1)[1] == 1) & (gold == 0)).float().sum()
        fn = ((scores.max(1)[1] == 0) & (gold == 1)).float().sum()
        return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()


class AccuracyMetric(Metric, name='accuracy'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        return (scores.max(1)[1] == gold).float().mean()


class RecallMetric(Metric, name='recall'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        return (((scores > 0) & (gold > 0)).float().sum(-1) / gold.float().sum(-1)).mean()


class F1Metric(Metric, name='f1'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        recall = (((scores > 0) & (gold > 0)).float().sum(-1) / gold.float().sum(-1)).mean()
        sum_scores = scores.float().sum(-1)
        sum_scores[sum_scores == 0] = 1
        prec = (((scores > 0) & (gold > 0)).float().sum(-1) / sum_scores).mean()
        return 2 * prec * recall / (prec + recall)


class PrecisionMetric(Metric, name='precision'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        sum_scores = scores.float().sum(-1)
        sum_scores[sum_scores == 0] = 1
        prec = (((scores > 0) & (gold > 0)).float().sum(-1) / sum_scores).mean()
        return prec


class PearsonrMetric(Metric, name='pearsonr'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        return torch.Tensor([pearsonr(scores.squeeze().tolist(), gold.squeeze().tolist())[0]])


class SpearmanrMetric(Metric, name='spearmanr'):

    def _evaluate(self, scores: torch.Tensor, gold: torch.Tensor):
        return torch.Tensor([spearmanr(scores.squeeze().tolist(), gold.squeeze().tolist())[0]])
