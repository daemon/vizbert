from dataclasses import dataclass
from typing import Sequence

from conllu import parse_incr
from conllu.models import TokenList
from transformers import PreTrainedTokenizer
import torch.utils.data as tud
import torch

from vizbert.utils import compute_distance_matrix, merge_by_segmentation, compute_coloring


class ConllDataset(tud.Dataset):

    def __init__(self, examples: Sequence[TokenList], sort_by_length=False):
        self.examples = examples
        if sort_by_length:
            self.examples = sorted(examples, key=len, reverse=True)
        self.attachables = {}

    def attach(self, name, data_list):
        assert len(data_list) == len(self.examples)
        self.attachables[name] = data_list

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.attachables:
            return ex, {k: v[idx] for k, v in self.attachables.items()}
        return ex

    @classmethod
    def from_file(cls, filename, **kwargs):
        with open(filename) as f:
            return cls(list(parse_incr(f)), **kwargs)


@dataclass
class TextBatch(object):
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    raw_text: Sequence[str]

    def pin_memory(self):
        self.token_ids.pin_memory()
        self.attention_mask.pin_memory()
        return self


@dataclass
class TreeDistanceBatch(object):
    hidden_states: torch.Tensor
    distance_matrix: torch.Tensor
    mask: torch.Tensor
    token_list_list: Sequence[TokenList]

    def pin_memory(self):
        self.hidden_states.pin_memory()
        self.distance_matrix.pin_memory()
        self.mask.pin_memory()
        return self


@dataclass
class ConllTextCollator(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples: Sequence[TokenList]):
        token_ids_lst = []
        raw_texts = []
        for ex in examples:
            sentence = ' '.join([x['form'] for x in ex])
            token_ids_lst.append(self.tokenizer.encode(sentence))
            raw_texts.append(sentence)
        max_len = max(map(len, token_ids_lst))
        attn_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in token_ids_lst])
        token_ids = torch.tensor([x + [0] * (max_len - len(x)) for x in token_ids_lst])
        return TextBatch(token_ids, attn_mask, raw_texts)


@dataclass
class ConllDistanceCollator(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples):
        hid_lst = []
        matrices = []
        token_list_list = []
        for ex, attached_data in examples:
            hid = attached_data['hidden_state']
            dist_matrix = compute_distance_matrix(ex)
            sentence = ' '.join([x['form'] for x in ex])
            coloring = compute_coloring(self.tokenizer, sentence)
            hid = merge_by_segmentation(hid, coloring).squeeze(1)
            hid_lst.append(hid)
            assert dist_matrix.size(0) == hid.size(0)
            matrices.append(dist_matrix)
            token_list_list.append(ex)
        max_len = max(x.size(0) for x in hid_lst)
        hid_tensor = torch.stack([torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in hid_lst])
        masks = [torch.ones_like(x) for x in matrices]
        dist_tensor = [torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in matrices]
        dist_tensor = torch.stack([torch.cat((x, torch.zeros(x.size(0), max_len - x.size(1))), 1) for x in dist_tensor])
        mask_tensor = [torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in masks]
        mask_tensor = torch.stack([torch.cat((x, torch.zeros(x.size(0), max_len - x.size(1))), 1) for x in mask_tensor])
        return TreeDistanceBatch(hid_tensor, dist_tensor, mask_tensor, token_list_list)
