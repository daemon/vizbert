from itertools import chain
from typing import Sequence, Union

from conllu.models import TokenList, TokenTree
from scipy.stats import spearmanr
import torch
import numpy as np

from .id_wrap import id_wrap


_SF_PUNCT = {"''", ',', '.', ':', '``', '-LRB-', '-RRB-'}


def compute_uuas(pred_tree: TokenTree,
                 gold_tree: TokenTree,
                 reduce=True,
                 ignore_punct=True):
    x1 = compute_distance_matrix(pred_tree)
    x2 = compute_distance_matrix(gold_tree)
    x2 = x2[:x1.size(0), :x1.size(1)]
    gold_tokens = flatten_tree(gold_tree)
    x1[x1 != 1] = 0
    x2[x2 != 1] = 0

    if ignore_punct:
        punct_ids = [tok['id'] for tok in gold_tokens if is_punctuation(tok['form']) and tok['id'] <= x2.size(0)]
        for punct_id in punct_ids:
            x2[:, punct_id - 1] = 0
            x2[punct_id - 1, :] = 0

    num = (x2 * (x1 == x2).float()).sum()
    if reduce:
        return (num / x2.sum()).item()
    return int(num.item()) // 2, int(x2.sum().item()) // 2


def compute_dspr(pred_matrix, gold_tree: TokenTree):
    gold_vec = compute_distance_matrix(gold_tree).contiguous().view(-1)
    pred_vec = pred_matrix.contiguous().view(-1)
    pred_vec = pred_vec[gold_vec != 0]
    gold_vec = gold_vec[gold_vec != 0]
    return spearmanr(pred_vec.tolist(), gold_vec.tolist())[0]


def is_punctuation(token, style='stanford'):
    if style == 'stanford':
        return token in _SF_PUNCT
    raise NotImplementedError


def flatten_tree(tree: TokenTree, token_list=None):
    if token_list is None:
        token_list = []
    token_list.append(tree.token)
    for child in tree.children:
        flatten_tree(child, token_list)
    return token_list


def compute_mst(distance_matrix: torch.Tensor,
                tokens: TokenList,
                ignore_punct=True) -> TokenTree:
    open_set = set([id_wrap(x) for x in tokens.copy()])
    closed_set = set()
    if ignore_punct:
        open_set = open_set - set(list(filter(lambda x: is_punctuation(x.obj['form']), open_set)))
    treenodes = {}
    root = None
    while open_set:
        if not closed_set:
            token = open_set.pop().obj
            treenodes[token['id']] = root = TokenTree(token, [])
            closed_set.add(id_wrap(token))
            continue
        grow_node_from = None
        grow_node_to = None
        grow_dist = np.inf
        for onode in open_set:
            onode = onode.obj
            for cnode in closed_set:
                cnode = cnode.obj
                dist = distance_matrix[onode['id'] - 1, cnode['id'] - 1]
                if dist < grow_dist:
                    grow_dist = dist
                    grow_node_from = cnode
                    grow_node_to = onode
        treenodes[grow_node_to['id']] = node = TokenTree(grow_node_to, [])
        treenodes[grow_node_from['id']].children.append(node)
        closed_set.add(id_wrap(grow_node_to))
        open_set.remove(id_wrap(grow_node_to))
    return root


def tree_length(tree: TokenTree):
    return 1 + sum(map(tree_length, tree.children))


def compute_distance_matrix(tokens: Union[TokenList, TokenTree]) -> torch.Tensor:
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

    if isinstance(tokens, TokenList):
        length = max(tok['id'] for tok in tokens)
        tree = tokens.to_tree()
    else:
        length = max(tok['id'] for tok in flatten_tree(tokens))
        tree = tokens
    distance_matrix = torch.zeros(length, length)
    set_parent_attr(tree)
    compute_pairwise_distances(tree)
    del_parent_attr(tree)
    return distance_matrix


def merge_by_segmentation(hidden_states, coloring, op=torch.mean):
    merged_states = []
    buffer = []
    last_color = 1 - coloring[0]
    for state, color in zip(hidden_states.split(1, 0), coloring):
        if last_color == color:
            buffer.append(state)
        else:
            if len(buffer) > 0:
                merged_states.append(op(torch.stack(buffer, 0), 0))
            buffer = [state]
        last_color = color
    if len(buffer) > 0:
        merged_states.append(op(torch.stack(buffer, 0), 0))
    return torch.cat(merged_states)


def compute_coloring(tokenizer, sentence) -> Sequence[int]:
    coloring = []
    last_color = 0
    for word in sentence.split(' '):
        coloring.extend([last_color] * len(tokenizer.tokenize(f' {word}')))
        last_color = 1 - last_color
    return coloring
