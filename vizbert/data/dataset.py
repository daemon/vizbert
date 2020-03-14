from itertools import chain
from dataclasses import dataclass
from typing import Sequence

from conllu import parse_incr
from conllu.models import TokenList
from transformers import PreTrainedTokenizer
import torch.utils.data as tud
import torch

from vizbert.utils import id_wrap


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


def compute_distance_matrix(tokens: TokenList):
    def set_parent_attr(node):
        for child in node.children:
            child.token['parent'] = node
            set_parent_attr(child)

    def del_parent_attr(node):
        for child in node.children:
            del child.token['parent']
            del_parent_attr(child)

    def compute_distances(source_node):
        closed_set = set()
        open_set = [(0, source_node)]
        while open_set:
            dist, node = open_set.pop()
            closed_set.add(id_wrap(node))
            i = node.token['id'] - 1
            j = source_node.token['id'] - 1
            distance_matrix[j, i] = distance_matrix[i, j] = dist
            for neighbor in chain(node.children, [node.token['parent']] if 'parent' in node.token else []):
                if id_wrap(neighbor) not in closed_set:
                    open_set.append((dist + 1, neighbor))

    def compute_pairwise_distances(curr_node):
        compute_distances(curr_node)
        for child in curr_node.children:
            compute_pairwise_distances(child)

    distance_matrix = torch.zeros(len(tokens), len(tokens))
    tree = tokens.to_tree()
    set_parent_attr(tree)
    compute_pairwise_distances(tree)
    del_parent_attr(tree)
    return distance_matrix


def merge_states(hidden_states, coloring):
    merged_states = []
    buffer = []
    last_color = 1 - coloring[0]
    for state, color in zip(hidden_states.split(1, 0), coloring):
        if last_color == color:
            buffer.append(state)
        else:
            if len(buffer) > 0:
                merged_states.append(torch.mean(torch.stack(buffer, 0), 0))
            buffer = [state]
        last_color = color
    if len(buffer) > 0:
        merged_states.append(torch.mean(torch.stack(buffer, 0), 0))
    return torch.stack(merged_states)


@dataclass
class ConllDistanceCollator(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, examples):
        hid_lst = []
        matrices = []
        for ex, attached_data in examples:
            hid = attached_data['hidden_state']
            dist_matrix = compute_distance_matrix(ex)
            sentence = ' '.join([x['form'] for x in ex])
            tokens = self.tokenizer.tokenize(sentence)
            coloring = []
            last_color = 0
            for tok in tokens:
                if tok.startswith('Ġ') or tok.startswith('##'):
                    last_color = 1 - last_color
                coloring.append(last_color)
            hid = merge_states(hid, coloring).squeeze(1)
            hid_lst.append(hid)
            matrices.append(dist_matrix)
        max_len = max(x.size(0) for x in hid_lst)
        hid_tensor = torch.stack([torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in hid_lst])
        masks = [torch.ones_like(x) for x in matrices]
        dist_tensor = [torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in matrices]
        dist_tensor = torch.stack([torch.cat((x, torch.zeros(x.size(0), max_len - x.size(1))), 1) for x in dist_tensor])
        mask_tensor = [torch.cat((x, torch.zeros(max_len - x.size(0), x.size(1)))) for x in masks]
        mask_tensor = torch.stack([torch.cat((x, torch.zeros(x.size(0), max_len - x.size(1))), 1) for x in mask_tensor])
        return TreeDistanceBatch(hid_tensor, dist_tensor, mask_tensor)
