from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.utils.data as tud
import torch

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import DATA_WORKSPACE_CLASSES, ClassificationCollator, TrainingWorkspace, LabeledTextBatch
from vizbert.inject import ModelInjector, ProbeDirectionRemovalModule, BertHiddenLayerInjectionHook
from vizbert.model import ModelTrainer, LOSS_SIZE_KEY, LOSS_KEY, ClassificationLoss


def main():
    def feeder(trainer, batch: LabeledTextBatch):
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids,
                       attention_mask=batch.attention_mask.to(args.device),
                       token_type_ids=batch.segment_ids.to(args.device))[0]
        labels = batch.labels.to(scores.device).float()
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
            ret['f1'] = (2 * (ret['recall'] * ret['precision']) / (ret['precision'] + ret['recall']))
            ret['f1'][torch.isnan(ret['f1'])] = 0
            ret['recall'] = ret['recall'].mean()
            ret['f1'] = ret['f1'].mean()
            ret['precision'] = ret['precision'].mean()
        else:
            ret['accuracy'] = (scores.max(-1)[1] == labels).float().mean()
        return ret

    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LR.default(5e-5),
                 OptionEnum.BATCH_SIZE.default(32),
                 OptionEnum.TASK,
                 OptionEnum.NUM_EPOCHS.default(3),
                 OptionEnum.WORKSPACE,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.MODEL.default('bert-base-uncased'),
                 OptionEnum.DEVICE,
                 OptionEnum.LOAD_WEIGHTS,
                 OptionEnum.EVAL_ONLY,
                 OptionEnum.LAYER_IDX.required(False),
                 opt('--probe-path', type=str),
                 opt('--max-seq-len', '-msl', type=int, default=128),
                 opt('--num-warmup-steps', '-nws', type=int, default=0))
    args = apb.parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dws = DATA_WORKSPACE_CLASSES[args.task](args.data_folder)
    datasets = dws.load_splits()
    tws = TrainingWorkspace(args.workspace)
    collator = ClassificationCollator(tokenizer, multilabel=datasets[0].multilabel, max_length=args.max_seq_len)
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

    if args.load_weights:
        tws.load_model(model)

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
                           model,
                           tws,
                           optimizer,
                           args.num_epochs,
                           scheduler=scheduler,
                           train_feed_loss_callback=feeder)
    hooks = []
    if args.probe_path:
        weights = torch.load(args.probe_path)['probe'].t()
        module = ProbeDirectionRemovalModule(weights)
        module.to(args.device)
        hooks.append(BertHiddenLayerInjectionHook(module, args.layer_idx))
    injector = ModelInjector(model.bert, hooks=hooks)
    with injector:
        if args.eval_only:
            trainer.evaluate(trainer.dev_loader, header='Dev')
            trainer.evaluate(trainer.test_loader, header='Test')
        else:
            trainer.train()


if __name__ == '__main__':
    main()
