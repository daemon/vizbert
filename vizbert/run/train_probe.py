from pathlib import Path
import argparse

from transformers import AutoTokenizer
import torch
import torch.utils.data as tud

from vizbert.data import InnerProductProbe, ConllDistanceCollator, ConllDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path)
    parser.add_argument('--hid-file', '-hf', type=Path)
    parser.add_argument('--probe-length', type=int, default=768)
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--batch-size', '-bsz', type=int, default=1)
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

    data_loader = tud.DataLoader(ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator)
    for batch in data_loader:
        print(batch.distance_matrix)


if __name__ == '__main__':
    main()
