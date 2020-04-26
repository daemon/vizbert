from transformers import BertForSequenceClassification
import torch


__all__ = ['expand_bert_classifier', 'shrink_bert_classifier']


def expand_bert_classifier(model: BertForSequenceClassification, num_labels):
    old_num_labels = model.num_labels
    model.num_labels = num_labels
    num_features = model.classifier.weight.size(1)
    weight = model.classifier.weight
    bias = model.classifier.bias
    with torch.no_grad():
        weight.data = torch.cat((weight.data,
                                 torch.zeros(num_labels - old_num_labels, num_features).to(weight.device)), 0)
        bias.data = torch.cat((bias.data, torch.ones(num_labels - old_num_labels).to(bias.device) * -10000), 0)


def shrink_bert_classifier(model: BertForSequenceClassification, num_labels):
    model.num_labels = num_labels
    with torch.no_grad():
        model.classifier.weight.data = model.classifier.weight.data[:num_labels]
        model.classifier.bias.data = model.classifier.bias.data[:num_labels]