import multiprocessing as mp

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, AutoConfig
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import ConllTextCollator, TrainingWorkspace, ClassificationCollator, DATA_WORKSPACE_CLASSES
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook
from vizbert.model import ProjectionPursuitProbe, EntropyLoss, ModelTrainer, LOSS_KEY, LOSS_SIZE_KEY, MaskedConceptLoss


def main():
    def entropy_feeder(_, batch):
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids, attention_mask=batch.attention_mask.to(args.device))[0]
        neg_entropy = criterion(scores)
        model.zero_grad()
        return {LOSS_KEY: neg_entropy,
                LOSS_SIZE_KEY: token_ids.size(0),
                'entropy': -neg_entropy}

    def classification_concept_loss_feeder(_, batch):
        token_ids = batch.token_ids.to(args.device)
        recon_scores = model(token_ids,
                             attention_mask=batch.attention_mask.to(args.device),
                             token_type_ids=batch.segment_ids.to(args.device))[0]
        with torch.no_grad():
            gold_scores = model2(token_ids,
                                 attention_mask=batch.attention_mask.to(args.device),
                                 token_type_ids=batch.segment_ids.to(args.device))[0]
        mask = torch.zeros_like(gold_scores)
        mask[:, args.mask_class] = 1
        loss = criterion(recon_scores, gold_scores, mask.bool())
        model.zero_grad()
        return {LOSS_KEY: loss,
                LOSS_SIZE_KEY: token_ids.size(0)}

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
                 opt('--dataset', '-d', type=str, default='conll', choices=['conll', 'sst2', 'reuters']),
                 opt('--num-features', type=int, default=768),
                 opt('--objective', type=str, default='concept', choices=['concept', 'entropy']),
                 opt('--mask-class', type=int))
    args = apb.parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

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
        model.load_state_dict(torch.load(args.load_weights))
        model2.load_state_dict(torch.load(args.load_weights))
    probe = ProjectionPursuitProbe(args.num_features,
                                   rank=args.probe_rank,
                                   mask_first=args.dataset != 'conll').to(args.device)
    # probe.probe.data = torch.load('pca.pt')['b'].t().to(args.device)
    hook = BertHiddenLayerInjectionHook(probe, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[hook])
    workspace = TrainingWorkspace(args.workspace)
    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    criterion = EntropyLoss() if args.objective == 'entropy' else MaskedConceptLoss(multilabel=dev_ds.multilabel)
    feeder = entropy_feeder if args.objective == 'entropy' else classification_concept_loss_feeder
    if args.dataset == 'conll':
        collator = ConllTextCollator(tokenizer)
    else:
        collator = ClassificationCollator(tokenizer, max_length=128, multilabel=dev_ds.multilabel)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    model.eval()
    model2.eval()
    optimizer = Adam(filter(lambda x: x.requires_grad, probe.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)
    trainer = ModelTrainer((train_loader, dev_loader, test_loader),
                           probe,
                           workspace,
                           optimizer,
                           args.num_epochs,
                           feeder,
                           scheduler=scheduler)
    with injector:
        trainer.train(test=args.dataset != 'sst2')


if __name__ == '__main__':
    main()
