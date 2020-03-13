from pathlib import Path
import argparse

from transformers import GPT2Tokenizer, GPT2Model
import torch.utils.data as tud

from vizbert.data import ConllDataset, ConllTextCollator
from vizbert.extract import ModelStateExtractor, Gpt2HiddenStateExtractor, TextDatasetInputFeeder, \
    BufferedFileOutputSerializer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path, required=True)
    parser.add_argument('--output-file', '-o', type=Path, required=True)
    parser.add_argument('--layer-idx', '-l', type=int, required=True)
    parser.add_argument('--extract-model', type=str, default='gpt2', choices=['gpt2'])
    parser.add_argument('--batch-size', '-bsz', type=int, default=50)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.extract_model)
    model = GPT2Model.from_pretrained(args.extract_model)
    ds = ConllDataset.from_file(args.data_file)
    collator = ConllTextCollator(tokenizer)

    model.cuda()
    model.eval()
    loader = tud.DataLoader(ds, collate_fn=collator, batch_size=args.batch_size, pin_memory=True)
    feeder = TextDatasetInputFeeder(loader, device='cuda')
    serializer = BufferedFileOutputSerializer(args.output_file)
    extractor = Gpt2HiddenStateExtractor(args.layer_idx)
    state_extractor = ModelStateExtractor(model, feeder, serializer, output_extractors=[extractor])
    state_extractor.extract()


if __name__ == '__main__':
    main()
