from pathlib import Path
import argparse

from torch.optim import Adam
from tqdm import tqdm, trange
from transformers import AutoTokenizer
import torch
import torch.utils.data as tud

from vizbert.data import InnerProductProbe, ConllDistanceCollator, ConllDataset, DistanceMatrixLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path)
    parser.add_argument('--hid-file', '-hf', type=Path)
    parser.add_argument('--probe-length', type=int, default=768)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--batch-size', '-bsz', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=torch.device, default='cuda:0')
    args = parser.parse_args()

    probe = InnerProductProbe(args.probe_length)
    ds = ConllDataset.from_file(args.data_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    collator = ConllDistanceCollator(tokenizer)
    hidden_states = torch.load(args.hid_file)['hidden_state']
    hid_joined = []
    for hid in hidden_states:
        x = torch.split(hid, 1)
        x = [y.squeeze(0).permute(1, 0, 2).contiguous().view(y.size(2), -1) for y in x]
        hid_joined.extend(x)
    ds.attach('hidden_state', hid_joined)

    criterion = DistanceMatrixLoss()
    optimizer = Adam(list(filter(lambda x: x.requires_grad, probe.parameters())), lr=args.lr)
    data_loader = tud.DataLoader(ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator)

    probe.to(args.device)
    for _ in trange(args.num_epochs, position=0):
        probe.train()
        pbar = tqdm(data_loader, total=len(data_loader), position=1)
        for batch in pbar:
            scores = probe(batch.hidden_states.to(args.device))
            loss = criterion(scores, batch.distance_matrix.to(args.device), batch.mask.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(dict(loss=f'{loss.item():.3}'))


if __name__ == '__main__':
    main()
