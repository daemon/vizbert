import multiprocessing as mp

from matplotlib import pyplot as plt
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.data import ConllWorkspace, ConllTextCollator, TrainingWorkspace, GlueWorkspace, GlueCollator
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook, ProbeReportingModule
from vizbert.utils import compute_coloring, merge_by_segmentation, batch_gs_coeffs, orth_tensor


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LAYER_IDX,
                 OptionEnum.WORKSPACE,
                 OptionEnum.PROBE_RANK,
                 OptionEnum.MODEL,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 opt('--load-weights', type=str),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--dataset-type', '-dt', type=str, default='conll', choices=['conll', 'glue']))
    args = apb.parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    if args.dataset_type == 'conll':
        model = BertForMaskedLM.from_pretrained(args.model).to(args.device)
    elif args.dataset_type == 'glue':
        model = BertForSequenceClassification.from_pretrained(args.model).to(args.device)
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))
    reporting_module = ProbeReportingModule()
    hook = BertHiddenLayerInjectionHook(reporting_module, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[hook])
    workspace = TrainingWorkspace(args.workspace)
    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    if args.dataset_type == 'conll':
        dw = ConllWorkspace(args.data_folder)
        dev_ds, = dw.load_conll_splits(splits=('dev',))
        collator = ConllTextCollator(tokenizer)
    else:
        dw = GlueWorkspace(args.data_folder)
        dev_ds, = dw.load_sst2_splits(splits=('dev',))
        collator = GlueCollator(tokenizer)
    dev_loader = tud.DataLoader(dev_ds, shuffle=True, batch_size=1, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)
    model.eval()

    proj_matrix = orth_tensor(torch.load(str(workspace.model_path))['probe'].to(args.device))
    if proj_matrix.size(1) != 2:
        print('Error: unable to visualize for non-two-dimensional probes.')
        return
    with injector, torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            model(batch.token_ids.to(args.device), attention_mask=batch.attention_mask.to(args.device))
            data = reporting_module.buffer
            coloring = compute_coloring(tokenizer, batch.raw_text[0])
            assert len(coloring) + 2 == batch.token_ids.size(1)
            # data = merge_by_segmentation(data[0], coloring).unsqueeze(0)
            coeffs = batch_gs_coeffs(proj_matrix, data).squeeze().cpu().numpy().T  # sequence, coeff
            plt.scatter(*coeffs, c='b', s=1.5)
            for pair, token in zip(coeffs.T, [tokenizer.convert_ids_to_tokens(x) for x in batch.token_ids[0].tolist()]):  #('[CLS] ' + batch.raw_text[0] + ' [SEP]').split(' ')):
                plt.annotate(token, pair, size='x-small')
            if idx == 10:
                break
    plt.show()


if __name__ == '__main__':
    main()
