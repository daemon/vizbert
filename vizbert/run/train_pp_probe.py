from pathlib import Path
import multiprocessing as mp

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, AutoConfig
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import ConllTextCollator, TrainingWorkspace, ClassificationCollator, DATA_WORKSPACE_CLASSES, ZeroMeanTransform
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook, ProbeReportingModule
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
        mask = torch.tensor(args.mask_classes).to(gold_scores.device) if args.mask_classes else None
        loss = criterion(recon_scores, gold_scores, mask)
        l1_penalty = probe.l1_penalty(args.l1_penalty)
        loss += l1_penalty
        model.zero_grad()
        return {LOSS_KEY: loss,
                LOSS_SIZE_KEY: token_ids.size(0),
                'l1': l1_penalty}

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
                 OptionEnum.EVAL_ONLY,
                 OptionEnum.USE_ZMT,
                 OptionEnum.OPTIMIZE_MEAN,
                 OptionEnum.DATASET,
                 OptionEnum.MAX_SEQ_LEN,
                 OptionEnum.INVERSE,
                 opt('--no-mask-first', action='store_false', dest='mask_first'),
                 opt('--opt-limit', type=int),
                 opt('--zmt-limit', type=int, default=1000),
                 opt('--load-probe', action='store_true'),
                 opt('--load-weights', type=Path),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--num-features', type=int, default=768),
                 opt('--objective', type=str, default='concept', choices=['concept', 'entropy']),
                 opt('--mask-classes', type=int, nargs='+', default=[]),
                 opt('--l1-penalty', type=float, default=0),
                 opt('--mask-weight', type=float, default=1))
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
        model.load_state_dict(torch.load(str(args.load_weights / 'model.pt')))
        model2.load_state_dict(torch.load(str(args.load_weights / 'model.pt')))
    probe = ProjectionPursuitProbe(args.num_features,
                                   rank=args.probe_rank,
                                   mask_first=args.mask_first,
                                   optimize_mean=args.optimize_mean,
                                   inverse=args.inverse).to(args.device)
    # probe.probe.data = torch.load('pca.pt')['b'].t().to(args.device)
    hook = BertHiddenLayerInjectionHook(probe, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[hook])
    workspace = TrainingWorkspace(args.workspace)
    if args.load_probe:
        sd = workspace.load_model(probe)
        for _, param in zip((k for k in sd if k.startswith('probe_params')), probe.probe_params):
            param.requires_grad = False

    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    criterion = EntropyLoss() if args.objective == 'entropy' else MaskedConceptLoss(multilabel=dev_ds.multilabel,
                                                                                    weight=args.mask_weight,
                                                                                    inverse=args.inverse)
    feeder = entropy_feeder if args.objective == 'entropy' else classification_concept_loss_feeder
    if args.dataset == 'conll':
        collator = ConllTextCollator(tokenizer)
    else:
        collator = ClassificationCollator(tokenizer, max_length=args.max_seq_len, multilabel=dev_ds.multilabel)
    train_loader = tud.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, shuffle=True, num_workers=args.num_workers)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    test_loader = tud.DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    model.eval()
    model2.eval()
    if args.use_zmt:
        zmt = ZeroMeanTransform(768, 2).to(args.device)
        reporting_module = ProbeReportingModule()
        t_hook = BertHiddenLayerInjectionHook(reporting_module, args.layer_idx - 1)
        zmt_injector = ModelInjector(model.bert, hooks=[t_hook])
        with zmt_injector:
            for idx, batch in enumerate(tqdm(train_loader, total=min(len(train_loader), args.zmt_limit), desc='Computing ZMT')):
                model(batch.token_ids.to(args.device),
                      attention_mask=batch.attention_mask.to(args.device),
                      token_type_ids=batch.segment_ids.to(args.device))
                zmt.update(reporting_module.buffer, mask=batch.attention_mask.to(args.device).unsqueeze(-1).expand_as(reporting_module.buffer))
                if idx == args.zmt_limit:
                    break
        probe.zmt = zmt

    optimizer = Adam(filter(lambda x: x.requires_grad, probe.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0)
    trainer = ModelTrainer((train_loader, dev_loader, test_loader),
                           probe,
                           workspace,
                           optimizer,
                           args.num_epochs,
                           feeder,
                           optimization_limit=args.opt_limit,
                           scheduler=scheduler)
    if args.eval_only:
        with injector:
            trainer.evaluate(trainer.dev_loader, 'Dev')
            trainer.evaluate(trainer.test_loader, 'Test')
    else:
        with injector:
            trainer.train(test=args.dataset != 'sst2')


if __name__ == '__main__':
    main()
