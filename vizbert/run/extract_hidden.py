from pathlib import Path
import multiprocessing as mp
import sys

from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, AutoConfig
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import TrainingWorkspace, ClassificationCollator, DATA_WORKSPACE_CLASSES
from vizbert.model import ModelTrainer, LOSS_KEY, LOSS_SIZE_KEY, ReconstructionLoss, LowRankProjectionTransform


def train(args):
    def forward_hook(module, input, output):
        hidden_states.append(output.cpu().detach())

    dw = DATA_WORKSPACE_CLASSES[args.dataset](args.data_folder)
    train_ds, dev_ds, test_ds = dw.load_splits()
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = train_ds.num_labels
    model = BertForSequenceClassification.from_pretrained(args.model, config=config).to(args.device)
    model.load_state_dict(torch.load(str(args.workspace / 'model.pt')))

    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    hidden_states = []
    outputs = []
    layers = model.bert.encoder.layer
    layer = layers[args.layer_idx - 1]
    layer.output.register_forward_hook(forward_hook)

    collator = ClassificationCollator(tokenizer, max_length=args.max_seq_len, multilabel=dev_ds.multilabel)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator,
                                  shuffle=True, num_workers=args.num_workers)
    model.eval()
    for batch in tqdm(train_loader, total=len(train_loader)):
        token_ids = batch.token_ids.to(args.device)
        outputs.append(model(token_ids,
                             attention_mask=batch.attention_mask.to(args.device),
                             token_type_ids=batch.segment_ids.to(args.device))[0].cpu().detach())
    torch.save((hidden_states, outputs), args.output_file)


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.MODEL,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 OptionEnum.BATCH_SIZE,
                 OptionEnum.DATASET,
                 OptionEnum.MAX_SEQ_LEN,
                 OptionEnum.LAYER_IDX,
                 opt('--output-file', '-o', type=str, required=True),
                 opt('--checkpoint', type=str, default='checkpoint.pt'),
                 opt('--workspace', '-w', type=Path, required=True),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--num-features', type=int, default=768))

    args = apb.parser.parse_args()
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    train(args)


if __name__ == '__main__':
    main()
