from pathlib import Path
import argparse

from transformers import GPT2Tokenizer, GPT2Model
import torch.utils.data as tud

from vizbert.data import ConllDataset, ConllTextCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path)
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2'])
    parser.add_argument('--batch-size', '-bsz', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model)
    ds = ConllDataset.from_file(args.data_file)
    collator = ConllTextCollator(tokenizer)
    loader = tud.DataLoader(ds, collate_fn=collator, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    model.cuda()
    model.config.output_hidden_states = True
    for batch in loader:
        x = model(batch.token_ids.cuda(), attention_mask=batch.attention_mask.cuda())
        print(x)


if __name__ == '__main__':
    main()
