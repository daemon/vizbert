from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.utils.data as tud
import torch

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import DATA_WORKSPACE_CLASSES, ClassificationCollator, TrainingWorkspace, LabeledTextBatch, ZeroMeanTransform
from vizbert.inject import ModelInjector, ProbeDirectionRemovalModule, BertHiddenLayerInjectionHook
from vizbert.model import ModelTrainer, LOSS_SIZE_KEY, LOSS_KEY, ClassificationLoss, ProjectionPursuitProbe


def main():
    def feeder(trainer, batch: LabeledTextBatch):
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids,
                       attention_mask=batch.attention_mask.to(args.device),
                       token_type_ids=batch.segment_ids.to(args.device))[0]
        labels = batch.labels.to(scores.device)
        if args.filter_labels:
            if batch.multilabel:
                mask = [any([idx in args.filter_labels for idx, lbl in enumerate(lbls) if lbl == 1]) for lbls in labels.tolist()]
            else:
                mask = [x in args.filter_labels for x in labels.tolist()]
            if args.invert_filter:
                mask = [not x for x in mask]
            scores = scores[mask]
            labels = labels[mask]
            token_ids = token_ids[mask]
        if token_ids.size(0) == 0:
            loss = torch.zeros(1).to(token_ids.device)
        else:
            loss = criterion(scores, labels)
        model.zero_grad()
        ret = {LOSS_KEY: loss,
               LOSS_SIZE_KEY: token_ids.size(0)}
        if trainer.training:
            scheduler.step()
        if batch.multilabel:
            ret['recall'] = ((scores > 0) & (labels > 0)).float().sum(-1) / labels.float().sum(-1)
            num_nnz_scores = (scores > 0).float().sum(-1)
            num_nnz_scores[num_nnz_scores == 0] = 1
            ret['precision'] = ((scores > 0) & (labels > 0)).float().sum(-1) / num_nnz_scores
            ret['recall'] = ret['recall'].mean()
            ret['precision'] = ret['precision'].mean()
            if token_ids.size(0) == 0:
                ret['precision'].zero_()
                ret['f1'].zero_()
                ret['recall'].zero_()
        else:
            if token_ids.size(0) == 0:
                ret['accuracy'] = torch.zeros(1).to(token_ids.device)
            else:
                ret['accuracy'] = (scores.max(-1)[1] == labels).float().mean()
        return ret

    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LR.default(5e-5),
                 OptionEnum.OPTIMIZE_MEAN,
                 OptionEnum.BATCH_SIZE.default(32),
                 OptionEnum.TASK,
                 OptionEnum.NUM_EPOCHS.default(3),
                 OptionEnum.WORKSPACE,
                 OptionEnum.PROBE_RANK.required(False),
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.MODEL.default('bert-base-uncased'),
                 OptionEnum.DEVICE,
                 OptionEnum.LOAD_WEIGHTS,
                 OptionEnum.EVAL_ONLY,
                 OptionEnum.USE_ZMT,
                 OptionEnum.LAYER_IDX.required(False),
                 opt('--probe-path', type=Path),
                 opt('--opt-limit', type=int),
                 opt('--tune-probe-weights', action='store_true'),
                 opt('--max-seq-len', '-msl', type=int, default=128),
                 opt('--num-warmup-steps', '-nws', type=int, default=0),
                 opt('--filter-labels', type=int, nargs='+', default=[]),
                 opt('--invert-filter', action='store_true'))
    args = apb.parser.parse_args()
    args.filter_labels = set(args.filter_labels)

    hooks = []
    if args.probe_path:
        probe = ProjectionPursuitProbe(768, args.probe_rank, mask_first=True, optimize_mean=args.optimize_mean)
        pws = TrainingWorkspace(args.probe_path)
        if args.use_zmt:
            zmt = ZeroMeanTransform(768, 2)
            probe.zmt = zmt
        try:
            pws.load_model(probe)
        except:
            pass
        # w = torch.load('svd.pt')
        # with torch.no_grad():
        #     probe.probe_params[0].set_(torch.from_numpy(w)[0])
        #     probe.probe_params[1].set_(torch.from_numpy(w)[1])
        probe.to(args.device)
        hooks.append(BertHiddenLayerInjectionHook(probe, args.layer_idx - 1))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dws = DATA_WORKSPACE_CLASSES[args.task](args.data_folder)
    datasets = dws.load_splits()
    tws = TrainingWorkspace(args.workspace)
    collator = ClassificationCollator(tokenizer,
                                      multilabel=datasets[0].multilabel,
                                      max_length=args.max_seq_len)
    loaders = [tud.DataLoader(ds,
                              batch_size=args.batch_size,
                              shuffle=do_shuffle,
                              num_workers=args.num_workers,
                              collate_fn=collator,
                              pin_memory=True) for ds, do_shuffle in zip(datasets, (True, False, False))]

    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = datasets[0].num_labels
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    model.to(args.device)
    injector = ModelInjector(model.bert, hooks=hooks)

    if args.load_weights:
        tws.load_model(model)

    if args.tune_probe_weights:
        optimizer = AdamW(list(filter(lambda x: x.requires_grad, probe.parameters())), lr=args.lr, eps=1e-8)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
            },
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    t_total = len(loaders[0]) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)
    criterion = ClassificationLoss(multilabel=datasets[0].multilabel)

    trainer = ModelTrainer(loaders,
                           probe if args.tune_probe_weights else model,
                           tws,
                           optimizer,
                           args.num_epochs,
                           scheduler=scheduler,
                           train_feed_loss_callback=feeder,
                           optimization_limit=args.opt_limit)
    with injector:
        if args.eval_only:
            trainer.evaluate(trainer.dev_loader, header='Dev')
            if datasets[2].labeled:
                trainer.evaluate(trainer.test_loader, header='Test')
        else:
            trainer.train(test=datasets[2].labeled)


if __name__ == '__main__':
    main()
