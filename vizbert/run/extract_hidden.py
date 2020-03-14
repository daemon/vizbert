from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.utils.data as tud

from vizbert.data import ConllTextCollator, DataWorkspace
from vizbert.extract import ModelStateExtractor, Gpt2HiddenStateExtractor, TextDatasetInputFeeder, \
    BufferedFileOutputSerializer, BertHiddenStateExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path, required=True)
    parser.add_argument('--layer-idx', '-l', type=int, required=True)
    parser.add_argument('--extract-model', type=str, default='gpt2', choices=['gpt2', 'bert-base-uncased'])
    parser.add_argument('--batch-size', '-bsz', type=int, default=50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    config = AutoConfig.from_pretrained(args.extract_model)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(args.extract_model, config=config)

    workspace = DataWorkspace(args.folder)
    train_ds, dev_ds, test_ds = workspace.load_conll_splits()
    collator = ConllTextCollator(tokenizer)

    model.cuda()
    model.eval()
    model.config.output_hidden = True
    for ds, name in zip((train_ds, dev_ds, test_ds), ('train.gold.conll', 'dev.gold.conll', 'test.gold.conll')):
        loader = tud.DataLoader(ds, collate_fn=collator, batch_size=args.batch_size, pin_memory=True)
        feeder = TextDatasetInputFeeder(loader, device='cuda')
        serializer = BufferedFileOutputSerializer(workspace.make_hidden_state_filename(name, args.layer_idx))
        if 'gpt2' in args.extract_model:
            extractor = Gpt2HiddenStateExtractor(args.layer_idx)
        elif 'bert' in args.extract_model:
            extractor = BertHiddenStateExtractor(args.layer_idx)
        state_extractor = ModelStateExtractor(model, feeder, serializer, output_extractors=[extractor])
        state_extractor.extract()


if __name__ == '__main__':
    main()
