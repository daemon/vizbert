import multiprocessing as mp

from matplotlib import pyplot as plt
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.model import ProjectionPursuitProbe
from vizbert.data import ConllWorkspace, ConllTextCollator, TrainingWorkspace, DATA_WORKSPACE_CLASSES, ClassificationCollator
from vizbert.inject import ModelInjector, BertHiddenLayerInjectionHook, ProbeReportingModule
from vizbert.utils import batch_gs_coeffs


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.LAYER_IDX,
                 OptionEnum.WORKSPACE,
                 OptionEnum.MODEL,
                 OptionEnum.NUM_WORKERS,
                 OptionEnum.DEVICE,
                 OptionEnum.PROBE_RANK,
                 OptionEnum.OUTPUT_FILE.required(False).default('output.pkl'),
                 opt('--project-type', type=str, choices=['pca', 'probe', 'tsne'], default='probe'),
                 opt('--load-weights', type=str),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--limit', type=int, default=30),
                 opt('--dataset', '-d', type=str, default='conll', choices=['conll', 'sst2', 'reuters', 'aapd', 'sst5']))
    args = apb.parser.parse_args()

    tok_config = {}
    if 'bert' in args.model:
        tok_config['do_basic_tokenize'] = args.basic_tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_config)
    if args.dataset == 'conll':
        dw = ConllWorkspace(args.data_folder)
        dev_ds, = dw.load_splits(splits=('dev',))
        collator = ConllTextCollator(tokenizer)
    else:
        dw = DATA_WORKSPACE_CLASSES[args.dataset](args.data_folder)
        dev_ds, = dw.load_splits(splits=('dev',))
        collator = ClassificationCollator(tokenizer, multilabel=dev_ds.multilabel, max_length=128)
    dev_loader = tud.DataLoader(dev_ds, shuffle=True, batch_size=1, pin_memory=True, collate_fn=collator, num_workers=args.num_workers)

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    if args.dataset == 'conll':
        model = BertForMaskedLM.from_pretrained(args.model).to(args.device)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model, num_labels=dev_ds.num_labels).to(args.device)
    if args.load_weights:
        model.load_state_dict(torch.load(args.load_weights))
    reporting_module = ProbeReportingModule()
    hook = BertHiddenLayerInjectionHook(reporting_module, args.layer_idx - 1)
    injector = ModelInjector(model.bert, hooks=[hook])
    workspace = TrainingWorkspace(args.workspace)
    model.eval()

    probe = ProjectionPursuitProbe(768, args.probe_rank)
    workspace.load_model(probe)
    proj_matrix = probe.orth_probe.to(args.device)
    if proj_matrix.size(1) > 3:
        print('Error: unable to visualize more than three dimensions.')
        return
    use_pca = args.project_type == 'pca'
    use_tsne = args.project_type == 'tsne'
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    if use_pca or use_tsne:
        states = []
        raw_texts = []
        colors = []
        with injector, torch.no_grad():
            for idx, batch in enumerate(dev_loader):
                scores = model(batch.token_ids.to(args.device), attention_mask=batch.attention_mask.to(args.device))
                data = reporting_module.buffer
                scores = scores[0].softmax(1)[0]
                colors.extend([(scores[0].item(), scores[1].item(), 0.0)] * (batch.token_ids.size(1) - 2))
                states.append(data[0][1:-1])
                raw_texts.extend([tokenizer.convert_ids_to_tokens(x) for x in batch.token_ids[0].tolist()[1:-1]])
                if idx == args.limit:
                    break
        states = torch.cat(states, 0).cpu().numpy()
        if use_pca:
            pca = PCA(2)
            pca.fit(states)
            torch.save(dict(b=torch.from_numpy(pca.components_)), 'pca.pt')
        elif use_tsne:
            tsne = TSNE()
            states = tsne.fit_transform(states)
            if args.probe_rank == 2:
                ax.scatter(*(states.T), c=colors, s=2)
            elif args.probe_rank == 3:
                ax.scatter3D(*(states.T), c=colors, s=2)
            for pair, token in zip(states, raw_texts):  #('[CLS] ' + batch.raw_text[0] + ' [SEP]').split(' ')):
                ax.text(*pair, token, size='xx-small')
            plt.show()
            return

    with injector, torch.no_grad():
        for idx, batch in enumerate(dev_loader):
            scores = model(batch.token_ids.to(args.device), attention_mask=batch.attention_mask.to(args.device))
            # scores = scores[0].softmax(1)[0]
            # colors = [(scores[0].item(), scores[1].item(), 0.0)] * batch.token_ids.size(1)
            data = reporting_module.buffer
            # coloring = compute_coloring(tokenizer, batch.raw_text[0])
            # assert len(coloring) + 2 == batch.token_ids.size(1)
            # data = merge_by_segmentation(data[0], coloring).unsqueeze(0)
            if use_pca:
                coeffs = pca.transform(data[0].cpu().numpy()).T
            else:
                coeffs = batch_gs_coeffs(proj_matrix, data).squeeze().cpu().numpy().T  # sequence, coeff
            if args.probe_rank == 2:
                ax.scatter(*coeffs, c='b', s=1.25)
            elif args.probe_rank == 3:
                ax.scatter3D(*coeffs, c='b', s=1.25)
            for pair, token in zip(coeffs.T, [tokenizer.convert_ids_to_tokens(x) for x in batch.token_ids[0].tolist()]):  #('[CLS] ' + batch.raw_text[0] + ' [SEP]').split(' ')):
                ax.text(*pair, token, size='xx-small')
            if idx == args.limit:
                break
    plt.show()


if __name__ == '__main__':
    main()
