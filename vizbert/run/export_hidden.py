from pathlib import Path
from tqdm import tqdm
import argparse
import pickle

from sklearn.decomposition import IncrementalPCA
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import torch
import torch.utils.data as tud

from vizbert.data import ConllDataset, ConllTextCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path, required=True)
    parser.add_argument('--output-file', '-o', type=Path, required=True)
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2'])
    parser.add_argument('--batch-size', '-bsz', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=50)
    parser.add_argument('--num-pca-components', '-npca', type=int, default=50)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model)
    ds = ConllDataset.from_file(args.data_file)
    collator = ConllTextCollator(tokenizer)
    loader = tud.DataLoader(ds, collate_fn=collator, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers, drop_last=True)

    model.cuda()
    model.eval()
    matrices = []
    raw_texts = []
    transform = IncrementalPCA(n_components=args.num_pca_components, batch_size=args.batch_size)
    max_cols = 2 * 12 * 64 * max(ex.token_ids.size(1) for ex in loader) * 12
    pbar = tqdm(loader, total=len(loader), desc='Fitting PCA')
    for batch in pbar:
        with torch.no_grad():
            x = model(batch.token_ids.cuda(), attention_mask=batch.attention_mask.cuda())
            x = torch.stack(x[1])
            x = x.permute(2, 0, 1, 3, 4, 5).contiguous()
            x = x.view(x.size(0), -1)
            transform.partial_fit(torch.cat((x.cpu(), torch.zeros(x.size(0), max_cols - x.size(1))), 1).numpy())
            raw_texts.extend(batch.raw_text)
            pbar.set_postfix(expvar=round(sum(transform.explained_variance_ratio_), 3))

    for batch in tqdm(loader, total=len(loader), desc='Writing matrices'):
        with torch.no_grad():
            x = model(batch.token_ids.cuda(), attention_mask=batch.attention_mask.cuda())
            x = torch.stack(x[1])
            x = x.permute(2, 0, 1, 3, 4, 5).contiguous()
            x = x.view(x.size(0), -1)
            x = torch.cat((x.cpu(), torch.zeros(x.size(0), max_cols - x.size(1))), 1)
            matrices.append(transform.transform(x.cpu().numpy()))

    matrix = np.vstack(matrices)
    tqdm.write(f'Final matrix shape: {matrix.shape}')
    with open(args.output_file, 'wb') as f:
        pickle.dump(dict(matrix=matrix, raw_texts=raw_texts), f, protocol=4)


if __name__ == '__main__':
    main()
