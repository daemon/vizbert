from pathlib import Path
import argparse
import multiprocessing as mp
import sys

from torch.optim import Adam
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.utils.data as tud

from vizbert.data import InnerProductProbe, ConllDistanceCollator, DataWorkspace, DistanceMatrixLoss
from vizbert.utils import compute_mst, compute_uuas


def main():
    def evaluate(loader, eval_uuas=False):
        probe.eval()
        score = 0
        tot = 0
        pbar = tqdm(loader, total=len(loader), position=1)
        with torch.no_grad():
            for batch in pbar:
                scores_batch = probe(batch.hidden_states.to(args.device))
                if eval_uuas:
                    scores_batch = scores_batch.cpu()
                    for scores, mask, tokenlist in zip(scores_batch, batch.mask, batch.token_list_list):
                        length = len(tokenlist)
                        scores = scores[:length, :length]
                        pred_tree = compute_mst(scores, tokenlist)
                        uuas = compute_uuas(pred_tree, tokenlist.to_tree())
                        if np.isnan(uuas):  # ignore single-word examples
                            continue
                        score += uuas
                        tot += 1
                    pbar.set_postfix(dict(uuas=f'{score / tot:.3}'))
                else:
                    loss = criterion(scores_batch, batch.distance_matrix.to(args.device), batch.mask.to(args.device))
                    score += loss.item() * scores_batch.size(0)
                    tot += scores_batch.size(0)
                    pbar.set_postfix(dict(loss=f'{score / tot:.3}'))
        return score / tot

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path)
    parser.add_argument('--probe-length', type=int, default=768)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--batch-size', '-bsz', type=int, default=20)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=torch.device, default='cuda:0')
    parser.add_argument('--layer-idx', '-l', type=int, default=6)
    parser.add_argument('--num-workers', type=int)
    args = parser.parse_args()

    probe = InnerProductProbe(args.probe_length)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    collator = ConllDistanceCollator(tokenizer)
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    print(f'Using {args.num_workers} workers.', file=sys.stderr)

    criterion = DistanceMatrixLoss()
    params = list(filter(lambda x: x.requires_grad, probe.parameters()))
    lr = args.lr
    optimizer = Adam(params, lr=lr)

    workspace = DataWorkspace(args.folder)
    train_ds, dev_ds, test_ds = workspace.load_conll_splits(attach_hidden=True, layer_idx=args.layer_idx)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    probe.to(args.device)
    best_dev_loss = 0
    for _ in trange(args.num_epochs, position=0):
        probe.train()
        pbar = tqdm(train_loader, total=len(train_loader), position=1)
        for idx, batch in enumerate(pbar):
            scores = probe(batch.hidden_states.to(args.device))
            loss = criterion(scores, batch.distance_matrix.to(args.device), batch.mask.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))
        dev_loss = evaluate(dev_loader)
        if dev_loss >= best_dev_loss:
            lr /= 10
            optimizer = Adam(params, lr=lr)
            best_dev_loss = dev_loss
    print(evaluate(test_loader, eval_uuas=True))


if __name__ == '__main__':
    main()
