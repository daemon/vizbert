from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.utils.data as tud

from vizbert.data import ConllTextCollator, ConllWorkspace
from vizbert.extract import ModelStateExtractor, Gpt2HiddenStateExtractor, TransformerInputFeeder, \
    BufferedFileOutputSerializer, BertHiddenStateExtractor, Gpt2AttentionKeyValueExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path, required=True)
    parser.add_argument('--layer-idx', '-l', type=int, required=True)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--batch-size', '-bsz', type=int, default=50)
    parser.add_argument('--do-basic-tokenize', action='store_true')
    parser.add_argument('--extract-type', type=str, default='hidden', choices=['hidden', 'key', 'value', 'keyvalue'])
    args = parser.parse_args()

    tok_config = dict()
    if 'bert' in args.model:
        tok_config = dict(do_basic_tokenize=args.do_basic_tokenize)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    config = AutoConfig.from_pretrained(args.model)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(args.model, config=config)

    workspace = ConllWorkspace(args.folder)
    train_ds, dev_ds, test_ds = workspace.load_conll_splits()
    collator = ConllTextCollator(tokenizer)

    model.cuda()
    model.eval()
    model.config.output_hidden = True
    for ds, name in zip((train_ds, dev_ds, test_ds), ('train.gold.conll', 'dev.gold.conll', 'test.gold.conll')):
        loader = tud.DataLoader(ds, collate_fn=collator, batch_size=args.batch_size, pin_memory=True)
        feeder = TransformerInputFeeder(loader, device='cuda')
        serializer = BufferedFileOutputSerializer(workspace.make_hidden_state_filename(name, args.layer_idx))
        if 'gpt2' in args.model and args.extract_type == 'hidden':
            extractor = Gpt2HiddenStateExtractor(args.layer_idx)
        elif 'bert' in args.model and args.extract_type == 'hidden':
            extractor = BertHiddenStateExtractor(args.layer_idx)
        elif 'gpt2' in args.model:
            extractor = Gpt2AttentionKeyValueExtractor(args.layer_idx, extract=args.extract_type)
        state_extractor = ModelStateExtractor(model, feeder, serializer, output_extractors=[extractor])
        state_extractor.extract()


if __name__ == '__main__':
    main()
