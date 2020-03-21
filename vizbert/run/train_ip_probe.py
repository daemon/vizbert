from pathlib import Path
import argparse
import multiprocessing as mp
import sys

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import numpy as np
import torch
import torch.utils.data as tud

from vizbert.data import ConllDistanceCollator, ConllWorkspace, TrainingWorkspace
from vizbert.model import InnerProductProbe, DistanceMatrixLoss
from vizbert.utils import compute_mst, compute_uuas, compute_dspr


def main():
    def evaluate(loader, test_eval=False):
        probe.eval()
        score = 0
        dspr_score = 0
        dspr_tot = 0
        uuas_tot = 0
        pbar = tqdm(loader, total=len(loader), position=1)
        with torch.no_grad():
            for batch in pbar:
                scores_batch = probe(batch.hidden_states.to(args.device))
                if test_eval:
                    scores_batch = scores_batch.cpu()
                    for scores, mask, tokenlist in zip(scores_batch, batch.mask, batch.token_list_list):
                        length = len(tokenlist)
                        scores = scores[:length, :length]
                        pred_tree = compute_mst(scores, tokenlist)
                        uuas, uuas_len = compute_uuas(pred_tree, tokenlist.to_tree(), reduce=False)
                        if np.isnan(uuas):  # ignore single-word examples
                            continue
                        dspr = compute_dspr(scores, tokenlist.to_tree())
                        if not np.isnan(dspr):
                            dspr_score += dspr
                            dspr_tot += 1
                        score += uuas
                        uuas_tot += uuas_len
                    pbar.set_postfix(dict(uuas=f'{score / uuas_tot:.3}', dspr=f'{dspr_score / dspr_tot:.3}'))
                else:
                    loss = criterion(scores_batch, batch.distance_matrix.to(args.device), batch.mask.to(args.device))
                    score += loss.item() * scores_batch.size(0)
                    uuas_tot += scores_batch.size(0)
                    pbar.set_postfix(dict(loss=f'{score / uuas_tot:.3}'))
        if test_eval:
            return score / uuas_tot, dspr_score / dspr_tot
        return score / uuas_tot

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', '-df', type=Path, required=True)
    parser.add_argument('--workspace', '-w', type=Path, required=True)
    parser.add_argument('--probe-length', type=int, default=768)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--batch-size', '-bsz', type=int, default=20)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=torch.device, default='cuda:0')
    parser.add_argument('--layer-idx', '-l', type=int, default=6)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--do-basic-tokenize', action='store_true')
    args = parser.parse_args()

    training_ws = TrainingWorkspace(args.workspace)
    probe = InnerProductProbe(args.probe_length)
    if args.load_model:
        training_ws.load_model(probe)
    probe.to(args.device)

    tok_config = dict()
    if 'bert' in args.tokenizer:
        tok_config = dict(do_basic_tokenize=args.do_basic_tokenize)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **tok_config)
    collator = ConllDistanceCollator(tokenizer)
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    print(f'Using {args.num_workers} workers.', file=sys.stderr)

    criterion = DistanceMatrixLoss()
    params = list(filter(lambda x: x.requires_grad, probe.parameters()))
    lr = args.lr
    optimizer = Adam(params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)

    data_ws = ConllWorkspace(args.data_folder)
    if args.eval_only:
        test_ds, = data_ws.load_splits(attach_hidden=True, layer_idx=args.layer_idx, splits=(data_ws.test_name,))
        test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
        uuas, dspr = evaluate(test_loader, test_eval=True)
        print(uuas, dspr)
        return

    train_ds, dev_ds, test_ds = data_ws.load_splits(attach_hidden=True, layer_idx=args.layer_idx)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    for epoch_idx in trange(args.num_epochs, position=0):
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
        scheduler.step(dev_loss)
        training_ws.summary_writer.add_scalar('Dev/Loss', dev_loss, epoch_idx)
        training_ws.save_model(probe)
    uuas, dspr = evaluate(test_loader, test_eval=True)
    print(uuas, dspr)
    training_ws.summary_writer.add_scalar('Test/UUAS', uuas)
    training_ws.summary_writer.add_scalar('Test/DSPR', dspr)


if __name__ == '__main__':
    main()
