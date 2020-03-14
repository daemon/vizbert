from sklearn.decomposition import IncrementalPCA

from .base import OutputTransform, OutputExtractor


class Gpt2PcaTransform(OutputTransform):

    def __init__(self, num_components, batch_size, pad_max_len=None):
        self.transform = IncrementalPCA(n_components=num_components, batch_size=batch_size)
        self.pad_max_len = pad_max_len

    def __call__(self, model, output):
        pass


class Gpt2HiddenStateExtractor(OutputExtractor):

    def __init__(self, layer_idx: int, last_only=True):
        self.layer_idx = layer_idx
        self.last_only = last_only

    def __call__(self, output):
        hidden = output[1][self.layer_idx]
        if self.last_only:
            hidden = hidden[-1]
        return dict(hidden_state=hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1).cpu().detach())


class BertHiddenStateExtractor(OutputExtractor):

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, output):
        hidden = output[2][self.layer_idx][:, 1:]
        return dict(hidden_state=hidden.cpu().detach())
