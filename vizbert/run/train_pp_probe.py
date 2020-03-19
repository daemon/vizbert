from pathlib import Path
import argparse
import multiprocessing as mp

from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, BertForMaskedLM
import torch.utils.data as tud
import torch

from vizbert.data import DataWorkspace, ConllTextCollator
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook
from vizbert.model import ProjectionPursuitProbe, EntropyLoss


def main():
    def evaluate(loader):
        tot_entropy = 0
        tot_len = 0
        with torch.no_grad():
            pbar = tqdm(loader, total=len(loader), position=1)
            for batch in pbar:
                token_ids = batch.token_ids.to(args.device)
                scores = model(token_ids, attention_mask=batch.attention_mask.to(args.device))[0]
                loss = criterion(scores)
                tot_entropy += loss.item() * batch.token_ids.size(0)
                tot_len += batch.token_ids.size(0)
                pbar.set_postfix(dict(H=f'{tot_entropy / tot_len:.3}'))
        return tot_entropy / tot_len

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', '-df', type=Path, required=True)
    parser.add_argument('--layer-idx', '-l', type=int, required=True)
    parser.add_argument('--num-features', type=int, default=768)
    parser.add_argument('--probe-rank', type=int, default=2)
    parser.add_argument('--model', default='bert-base-cased', type=str)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--device', type=torch.device, default='cuda:0')
    parser.add_argument('--batch-size', '-bsz', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num-epochs', type=int, default=10)
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    model = BertForMaskedLM.from_pretrained(args.model).to(args.device)
    probe = ProjectionPursuitProbe(args.num_features, rank=args.probe_rank).to(args.device)
    extractor = BertHiddenLayerInjectionHook(probe, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[extractor])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = ConllTextCollator(tokenizer)

    criterion = EntropyLoss()
    train_ds, dev_ds, test_ds = DataWorkspace(args.data_folder).load_conll_splits()
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    model.eval()
    optimizer = Adam(filter(lambda x: x.requires_grad, probe.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)

    with injector:
        for _ in trange(args.num_epochs, position=0):
            pbar = tqdm(train_loader, total=len(train_loader), position=1)
            for batch in pbar:
                token_ids = batch.token_ids.to(args.device)
                scores = model(token_ids, attention_mask=batch.attention_mask.to(args.device))[0]
                loss = criterion(scores)
                optimizer.zero_grad()
                loss.backward()
                pbar.set_postfix(dict(H=f'{loss.item():.3}'))
            dev_loss = evaluate(dev_loader)
            scheduler.step(dev_loss)
    print(evaluate(test_loader))


if __name__ == '__main__':
    main()
