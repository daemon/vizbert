from pathlib import Path
import multiprocessing as mp

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, AutoConfig
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import TrainingWorkspace, ClassificationCollator, DATA_WORKSPACE_CLASSES
from vizbert.inject import ModelInjector, BertAttentionMatrixKeyInjectionHook, BertAttentionMatrixQueryInjectionHook,\
    BertAttentionMatrixValueInjectionHook
from vizbert.model import ModelTrainer, LOSS_KEY, LOSS_SIZE_KEY, ReconstructionLoss, LowRankProjectionTransform


def train(args):
    def recon_feeder(_, batch):
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids, attention_mask=batch.attention_mask.to(args.device))[0]
        gold_scores = model2(token_ids,
                             attention_mask=batch.attention_mask.to(args.device),
                             token_type_ids=batch.segment_ids.to(args.device))[0]
        loss = criterion(scores, gold_scores)  # TODO: multilabel
        model.zero_grad()
        model2.zero_grad()
        return {LOSS_KEY: loss,
                LOSS_SIZE_KEY: token_ids.size(0)}

    dw = DATA_WORKSPACE_CLASSES[args.dataset](args.data_folder)
    train_ds, dev_ds, test_ds = dw.load_splits()
    if args.dataset == 'conll':
        model = BertForMaskedLM.from_pretrained(args.model).to(args.device)
        model2 = BertForMaskedLM.from_pretrained(args.model).to(args.device)
    else:
        config = AutoConfig.from_pretrained(args.model)
        config.num_labels = train_ds.num_labels
        model = BertForSequenceClassification.from_pretrained(args.model, config=config).to(args.device)
        model2 = BertForSequenceClassification.from_pretrained(args.model, config=config).to(args.device)
    if args.load_weights:
        model.load_state_dict(torch.load(str(args.load_weights / 'model.pt')))
        model2.load_state_dict(torch.load(str(args.load_weights / 'model.pt')))

    for rank in args.ranks:
        transform = LowRankProjectionTransform(args.num_features, rank).to(args.device)
        key_hook = BertAttentionMatrixKeyInjectionHook(transform, args.layer_idx - 1)
        value_hook = BertAttentionMatrixValueInjectionHook(transform, args.layer_idx - 1)
        query_hook = BertAttentionMatrixQueryInjectionHook(transform, args.layer_idx - 1)
        injector = ModelInjector(model.bert, hooks=(key_hook, value_hook, query_hook))
        workspace = TrainingWorkspace(Path(f'{args.workspace_prefix}-kqv-{args.dataset}-l{args.layer_idx}-r{rank}'))
        if args.load_probe:
            sd = workspace.load_model(transform)
            for _, param in zip((k for k in sd if k.startswith('probe_params')), transform.probe_params):
                param.requires_grad = False

        tok_config = {}
        if 'bert' in args.model:
            tok_config['do_basic_tokenize'] = args.basic_tokenize
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
        criterion = ReconstructionLoss(multilabel=dev_ds.multilabel)

        feeder = recon_feeder
        collator = ClassificationCollator(tokenizer, max_length=args.max_seq_len, multilabel=dev_ds.multilabel)
        train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator,
                                      shuffle=True, num_workers=args.num_workers)
        dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator,
                                    num_workers=args.num_workers)
        test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator,
                                     num_workers=args.num_workers)

        model.eval()
        model2.eval()
        optimizer = Adam(filter(lambda x: x.requires_grad, transform.parameters()), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)
        trainer = ModelTrainer((train_loader, dev_loader, test_loader),
                               transform,
                               workspace,
                               optimizer,
                               args.num_epochs,
                               feeder,
                               optimization_limit=args.opt_limit,
                               scheduler=scheduler)
        if not args.eval_only:
            with injector:
                trainer.train(test=args.dataset != 'sst2')
        trainer.evaluate(trainer.dev_loader, 'Dev')


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LAYER_IDX,
                 OptionEnum.PROBE_RANK,
                 OptionEnum.MODEL,
                 OptionEnum.NUM_EPOCHS,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 OptionEnum.BATCH_SIZE,
                 OptionEnum.LR,
                 OptionEnum.EVAL_ONLY,
                 OptionEnum.DATASET,
                 OptionEnum.MAX_SEQ_LEN,
                 opt('--workspace-prefix', type=str, required=True),
                 opt('--no-mask-first', action='store_false', dest='mask_first'),
                 opt('--opt-limit', type=int),
                 opt('--load-probe', action='store_true'),
                 opt('--load-weights', type=Path),
                 opt('--ranks', nargs='+', type=int, default=(1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 576)),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--num-features', type=int, default=768),
                 opt('--mask-classes', type=int, nargs='+', default=[]),
                 opt('--l1-penalty', type=float, default=0),
                 opt('--mask-weight', type=float, default=1))

    args = apb.parser.parse_args()
    if args.probe_rank:
        args.ranks = (args.probe_rank,)
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    train(args)


if __name__ == '__main__':
    main()
