from pathlib import Path
import multiprocessing as mp
import sys

from torch.optim import Adam
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification, AutoConfig
from transformers import get_linear_schedule_with_warmup
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.model import LowRankBertOutput, LowRankBertSelfAttention, LowRankBertIntermediate, LowRankBertSelfOutput
from vizbert.data import TrainingWorkspace, ClassificationCollator, DATA_WORKSPACE_CLASSES
from vizbert.model import ModelTrainer, LOSS_KEY, LOSS_SIZE_KEY, ReconstructionLoss, LowRankProjectionTransform


def train(args):
    def recon_feeder(_, batch):
        model.eval()
        model2.eval()
        token_ids = batch.token_ids.to(args.device)
        scores = model(token_ids,
                       attention_mask=batch.attention_mask.to(args.device),
                       token_type_ids=batch.segment_ids.to(args.device))[0]
        gold_scores = model2(token_ids,
                             attention_mask=batch.attention_mask.to(args.device),
                             token_type_ids=batch.segment_ids.to(args.device))[0]
        loss = criterion(scores, gold_scores)  # TODO: multilabel
        model.zero_grad()
        model2.zero_grad()
        if not trainer.training:
            all_scores.append(scores.detach())
            all_gold_scores.append(batch.labels.to(scores.device))
        if batch.labels is not None:
            accuracy = (batch.labels.to(scores.device) == scores.max(1)[1]).float().sum() / token_ids.size(0)
        else:
            accuracy = 0
        if trainer.training:
            scheduler.step()
        return {LOSS_KEY: loss,
                LOSS_SIZE_KEY: token_ids.size(0),
                'accuracy': accuracy}

    all_scores = []
    all_gold_scores = []

    dw = DATA_WORKSPACE_CLASSES[args.dataset](args.data_folder)
    train_ds, dev_ds, test_ds = dw.load_splits()
    ranks = [args.probe_rank] if args.probe_rank else [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 512]
    for rank in ranks:
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

        workspace = TrainingWorkspace(Path(f'{args.workspace_prefix}-kqv-{args.dataset}-r{rank}'))
        tok_config = {}
        if 'bert' in args.model:
            tok_config['do_basic_tokenize'] = args.basic_tokenize
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
        criterion = ReconstructionLoss(multilabel=dev_ds.multilabel)
        params = []
        layers = model.bert.encoder.layer
        layer = layers[args.layer_idx - 1]
        # for layer in layers[1:]:
        lro = LowRankBertOutput(model.config, rank).to(args.device).init_pretrained(layer.output)
        # lri = LowRankBertIntermediate(model.config, rank).to(args.device).init_pretrained(layer.intermediate)
        # lrsa = LowRankBertSelfAttention(model.config, rank).to(args.device).init_pretrained(layer.attention.self)
        # lrso = LowRankBertSelfOutput(model.config, rank).to(args.device).init_pretrained(layer.attention.output)

        layer.output = lro
        # layer.intermediate = lri
        # layer.attention.self = lrsa
        # layer.attention.output = lrso

        params.extend(lro.low_rank_parameters())
        # params.extend(lri.low_rank_parameters())
        # params.extend(lrsa.low_rank_parameters())
        # params.extend(lrso.low_rank_parameters())
        if args.load_svd:
            components, _ = torch.load(args.load_svd)
            print(components.shape)
            lro.low.orth_probe = torch.from_numpy(components)
        elif args.load_checkpoint:
            model3 = torch.load(args.checkpoint).to(args.device)
            lro.low = model3.bert.encoder.layer[args.from_layer_idx - 1].output.low.to(args.device)

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
        optimizer = Adam(params, lr=args.lr)
        steps = min(len(train_loader) * args.num_epochs, args.opt_limit)
        scheduler = get_linear_schedule_with_warmup(optimizer, steps // 10, steps)
        trainer = ModelTrainer((train_loader, dev_loader, test_loader),
                               model,
                               workspace,
                               optimizer,
                               args.num_epochs,
                               feeder,
                               optimization_limit=args.opt_limit)
        if not args.eval_only:
            trainer.train(test=test_ds.labeled)
        all_scores = []
        all_gold_scores = []
        trainer.evaluate(trainer.dev_loader, 'Dev')
        quality = train_ds.evaluate_metrics(torch.cat(all_scores), torch.cat(all_gold_scores))[args.quality_key].item()
        if not args.eval_only:
            torch.save(model, args.checkpoint)
            if quality > args.orig_quality * 0.95:
                print(f'found {rank}')
                break


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.PROBE_RANK.required(False),
                 OptionEnum.MODEL,
                 OptionEnum.NUM_EPOCHS,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 OptionEnum.BATCH_SIZE,
                 OptionEnum.LR,
                 OptionEnum.EVAL_ONLY,
                 OptionEnum.DATASET,
                 OptionEnum.MAX_SEQ_LEN,
                 OptionEnum.LAYER_IDX,
                 opt('--load-svd', type=str),
                 opt('--checkpoint', type=str, default='checkpoint.pt'),
                 opt('--from-layer-idx', type=int),
                 opt('--workspace-prefix', type=str, required=True),
                 opt('--no-mask-first', action='store_false', dest='mask_first'),
                 opt('--opt-limit', type=int, default=1000000),
                 opt('--load-probe', action='store_true'),
                 opt('--load-weights', type=Path),
                 opt('--orig-quality', '-oq', type=float, default=0),
                 opt('--quality-key', '-qk', type=str, default='accuracy'),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--num-features', type=int, default=768),
                 opt('--mask-classes', type=int, nargs='+', default=[]),
                 opt('--l1-penalty', type=float, default=0),
                 opt('--load-checkpoint', action='store_true'),
                 opt('--mask-weight', type=float, default=1))

    args = apb.parser.parse_args()
    if args.from_layer_idx is None:
        args.from_layer_idx = args.layer_idx
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
    train(args)


if __name__ == '__main__':
    main()
