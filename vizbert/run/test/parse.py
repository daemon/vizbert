from pathlib import Path
import argparse

from vizbert.data import DataWorkspace
from vizbert.utils import compute_mst, compute_uuas, compute_distance_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path)
    parser.add_argument('--tokenizer', type=str, default='gpt2', choices=['gpt2', 'bert-base-uncased'])
    parser.add_argument('--layer-idx', type=int, default=5)
    args = parser.parse_args()

    workspace = DataWorkspace(args.folder)
    dev_ds = workspace.load_attached_hidden_state_dataset(workspace.dev_name, args.layer_idx)
    tokenlist, batch = dev_ds[8]
    hid = batch['hidden_state']
    print([tok['id'] for tok in tokenlist])
    tokenlist.to_tree().print_tree()
    dm = compute_distance_matrix(tokenlist)
    print(dm)
    pred_tree = compute_mst(dm, tokenlist)
    pred_tree.print_tree()
    print(compute_uuas(pred_tree, tokenlist.to_tree(), reduce=False), len(tokenlist), dm.size())



if __name__ == '__main__':
    main()
