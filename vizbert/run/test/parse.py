from pathlib import Path
import argparse

from torch.optim import Adam
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import torch.utils.data as tud

from vizbert.data import InnerProductProbe, ConllDistanceCollator, DataWorkspace, DistanceMatrixLoss, ConllDataset
from vizbert.utils import compute_mst, compute_uuas, compute_coloring, compute_distance_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path)
    parser.add_argument('--tokenizer', type=str, default='gpt2', choices=['gpt2', 'bert-base-uncased'])
    parser.add_argument('--layer-idx', type=int, default=5)
    args = parser.parse_args()

    workspace = DataWorkspace(args.folder)
    dev_ds = workspace.load_attached_hidden_state_dataset(workspace.dev_name, args.layer_idx)
    tokenlist, hid = dev_ds[2]
    hid = hid['hidden_state']
    print([tok['id'] for tok in tokenlist])
    tokenlist.to_tree().print_tree()
    dm = compute_distance_matrix(tokenlist)
    pred_tree = compute_mst(dm, tokenlist)
    pred_tree.print_tree()
    print(compute_uuas(pred_tree, tokenlist.to_tree()))


if __name__ == '__main__':
    main()
