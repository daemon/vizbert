from pathlib import Path
from typing import Sequence

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers.modeling_bert import BertSelfAttention
import torch.utils.data as tud
import torch

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import DATA_WORKSPACE_CLASSES, ClassificationCollator, TrainingWorkspace, LabeledTextBatch, ZeroMeanTransform,\
    DataFrameDataset
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook, BertAttentionMatrixKeyInjectionHook,\
    BertAttentionMatrixValueInjectionHook, BertAttentionMatrixQueryInjectionHook, InstrumentModuleInjectionHook
from vizbert.model import ModelTrainer, LOSS_SIZE_KEY, LOSS_KEY, ClassificationLoss, LowRankProjectionTransform
from vizbert.utils import expand_bert_classifier


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
        ret.update(datasets[0].evaluate_metrics(scores, labels))
        if trainer.training:
            trainer.post_callbacks(scheduler.step)
        return ret

    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LR.default(5e-5),
                 OptionEnum.OPTIMIZE_MEAN,
                 OptionEnum.BATCH_SIZE.default(32),
                 OptionEnum.EVAL_BATCH_SIZE,
                 OptionEnum.DATASET,
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
                 OptionEnum.MAX_SEQ_LEN,
                 OptionEnum.INVERSE,
                 opt('--no-mask-first', action='store_false', dest='mask_first'),
                 opt('--expand-labels', type=int),
                 opt('--probe-path', type=Path),
                 opt('--opt-limit', type=int),
                 opt('--tune-probe-weights', action='store_true'),
                 opt('--num-warmup-steps', '-nws', type=int, default=0),
                 opt('--filter-labels', type=int, nargs='+', default=[]),
                 opt('--invert-filter', action='store_true'))
    args = apb.parser.parse_args()
    args.filter_labels = set(args.filter_labels)

    hooks = []
    if args.probe_path:
        probe = LowRankProjectionTransform(768, args.probe_rank)
        pws = TrainingWorkspace(args.probe_path)
        if args.use_zmt:
            zmt = ZeroMeanTransform(768, 2)
            probe.zmt = zmt
        try:
            pws.load_model(probe)
        except:
            pass
        probe.to(args.device)
        hooks.append(BertAttentionMatrixKeyInjectionHook(probe, args.layer_idx - 1))
        hooks.append(BertAttentionMatrixValueInjectionHook(probe, args.layer_idx - 1))
        hooks.append(BertAttentionMatrixQueryInjectionHook(probe, args.layer_idx - 1))
    module_hook = InstrumentModuleInjectionHook()
    hooks.append(module_hook)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dws = DATA_WORKSPACE_CLASSES[args.dataset](args.data_folder)
    datasets = dws.load_splits()  # type: Sequence[DataFrameDataset]
    tws = TrainingWorkspace(args.workspace)
    tws.write_args(args)
    collator = ClassificationCollator(tokenizer,
                                      multilabel=datasets[0].multilabel,
                                      max_length=args.max_seq_len)
    loaders = [tud.DataLoader(ds,
                              batch_size=bsz,
                              shuffle=do_shuffle,
                              num_workers=args.num_workers,
                              collate_fn=collator,
                              pin_memory=True) for ds, do_shuffle, bsz in
               zip(datasets, (True, False, False), (args.batch_size, args.eval_batch_size, args.eval_batch_size))]

    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = datasets[0].num_labels
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    model.to(args.device)
    injector = ModelInjector(model.bert, hooks=hooks)

    if args.load_weights:
        tws.load_model(model)
    if args.expand_labels:
        expand_bert_classifier(model, args.expand_labels)

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
            # if datasets[2].labeled:
            #     trainer.evaluate(trainer.test_loader, header='Test')
            timings = module_hook.timer.timings
            print(timings.keys())
            print(('encoder total', timings['encoder']),
                  ('pooler total', timings['pooler']),
                  sum(v for k, v in timings.items() if 'self.value' in k),
                  sum(v for k, v in timings.items() if 'self.query' in k),
                  sum(v for k, v in timings.items() if 'self.key' in k),
                  ('all attn', sum(v for k, v in timings.items() if k.endswith('attention.self'))),
                  ('odense', sum(v for k, v in timings.items() if k.endswith('output.dense'))),
                  ('idense', sum(v for k, v in timings.items() if k.endswith('intermediate.dense'))),
                  ('pdense', timings['pooler.dense']),
                  ('qkmm', BertSelfAttention.qkmm_time),
                  ('avmm', BertSelfAttention.avmm_time))
        else:
            trainer.train(test=datasets[2].labeled)


if __name__ == '__main__':
    main()
