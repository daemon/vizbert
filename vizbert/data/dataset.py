from dataclasses import dataclass
from typing import Sequence

from conllu import parse_incr
from conllu.models import TokenList
from transformers import PreTrainedTokenizer
import torch.utils.data as tud
import torch


class ConllDataset(tud.Dataset):

    def __init__(self, examples: Sequence[TokenList]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            return cls(list(parse_incr(f)))


@dataclass
class TextBatch(object):
    token_ids: torch.Tensor
    attention_mask: torch.Tensor

    def pin_memory(self):
        self.token_ids.pin_memory()
        self.attention_mask.pin_memory()
        return self


@dataclass
class ConllTextCollator(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples: Sequence[TokenList]):
        token_ids_lst = []
        for ex in examples:
            sentence = ' '.join([x['form'] for x in ex])
            token_ids_lst.append(self.tokenizer.encode(sentence))
        max_len = max(map(len, token_ids_lst))
        attn_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in token_ids_lst])
        token_ids = torch.tensor([x + [0] * (max_len - len(x)) for x in token_ids_lst])
        return TextBatch(token_ids, attn_mask)
