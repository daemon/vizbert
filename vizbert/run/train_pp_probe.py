import multiprocessing as mp

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import ConllWorkspace, ConllTextCollator, TrainingWorkspace, GlueWorkspace, GlueCollator
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook
from vizbert.model import ProjectionPursuitProbe, EntropyLoss, ModelTrainer, LOSS_KEY, LOSS_SIZE_KEY


def main():
    def entropy_feeder(_, batch):
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids, attention_mask=batch.attention_mask.to(args.device))[0]
        neg_entropy = criterion(scores)
        model.zero_grad()
        return {LOSS_KEY: neg_entropy,
                LOSS_SIZE_KEY: token_ids.size(0),
                'entropy': -neg_entropy}

    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LAYER_IDX,
                 OptionEnum.WORKSPACE,
                 OptionEnum.PROBE_RANK,
                 OptionEnum.MODEL,
                 OptionEnum.NUM_EPOCHS,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 OptionEnum.BATCH_SIZE,
                 OptionEnum.LR,
                 opt('--load-weights', type=str),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--dataset-type', '-dt', type=str, default='conll', choices=['conll', 'glue']),
                 opt('--num-features', type=int, default=768))
    args = apb.parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    if args.dataset_type == 'conll':
        model = BertForMaskedLM.from_pretrained(args.model).to(args.device)
    elif args.dataset_type == 'glue':
        model = BertForSequenceClassification.from_pretrained(args.model).to(args.device)
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))
    probe = ProjectionPursuitProbe(args.num_features,
                                   rank=args.probe_rank,
                                   mask_first=args.dataset_type == 'glue').to(args.device)
    # probe.probe.data = torch.load('pca.pt')['b'].t().to(args.device)
    hook = BertHiddenLayerInjectionHook(probe, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[hook])
    workspace = TrainingWorkspace(args.workspace)
    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    criterion = EntropyLoss()
    if args.dataset_type == 'conll':
        dw = ConllWorkspace(args.data_folder)
        train_ds, dev_ds, test_ds = dw.load_conll_splits()
        collator = ConllTextCollator(tokenizer)
    else:
        dw = GlueWorkspace(args.data_folder)
        train_ds, dev_ds, test_ds = dw.load_sst2_splits()
        collator = GlueCollator(tokenizer)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    model.eval()
    optimizer = Adam(filter(lambda x: x.requires_grad, probe.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)
    trainer = ModelTrainer((train_loader, dev_loader, test_loader),
                           probe,
                           workspace,
                           optimizer,
                           args.num_epochs,
                           entropy_feeder,
                           scheduler=scheduler)

    with injector:
        trainer.train(test=args.dataset_type != 'glue')


if __name__ == '__main__':
    main()
