from itertools import chain
from typing import Sequence, Union

from conllu.models import TokenList, TokenTree
import torch
import numpy as np

from .id_wrap import id_wrap


def compute_uuas(pred_tree: TokenTree, gold_tree: TokenTree):
    treelen1 = tree_length(pred_tree)
    treelen2 = tree_length(pred_tree)
    assert treelen1 == treelen2
    x1 = compute_distance_matrix(pred_tree)
    x2 = compute_distance_matrix(gold_tree)
    x1[x1 != 1] = 0
    x2[x2 != 1] = 0
    x1 = x1.view(-1)
    x2 = x2.view(-1)
    return ((x2 * (x1 == x2).float()).sum() / x2.sum()).item()


def compute_mst(distance_matrix: torch.Tensor, tokens: TokenList) -> TokenTree:
    open_set = set([id_wrap(x) for x in tokens.copy()])
    closed_set = set()
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
        length = len(tokens)
        tree = tokens.to_tree()
    else:
        length = tree_length(tokens)
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
