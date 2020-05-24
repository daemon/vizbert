from sklearn.decomposition import IncrementalPCA

from .base import OutputTransform, OutputExtractor


__all__ = ['Gpt2AttentionKeyValueExtractor', 'BertHiddenStateExtractor', 'Gpt2HiddenStateExtractor']


class Gpt2PcaTransform(OutputTransform):

    def __init__(self, num_components, batch_size, pad_max_len=None):
        self.transform = IncrementalPCA(n_components=num_components, batch_size=batch_size)
        self.pad_max_len = pad_max_len

    def __call__(self, model, output):
        pass


class Gpt2AttentionKeyValueExtractor(OutputExtractor):

    def __init__(self, layer_idx: int, extract='key'):
        self.layer_idx = layer_idx
        if extract == 'key':
            self.extract_idx = 0
        elif extract == 'value':
            self.extract_idx = 1
        else:
            self.extract_idx = (0, 1)

    def __call__(self, output):
        hidden = output[1][self.layer_idx][self.extract_idx]
        return dict(hidden_state=hidden.permute(0, 2, 1, 3).contiguous().view(hidden.size(0), hidden.size(2), -1).cpu().detach())


class Gpt2HiddenStateExtractor(OutputExtractor):

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, output):
        hidden = output[2][self.layer_idx]
        return dict(hidden_state=hidden.cpu().detach())


class BertHiddenStateExtractor(OutputExtractor):

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, output):
        hidden = output[2][self.layer_idx]
        return dict(hidden_state=hidden.cpu().detach())
