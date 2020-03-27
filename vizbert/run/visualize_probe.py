import multiprocessing as mp

from matplotlib import pyplot as plt
from transformers import AutoTokenizer, BertForMaskedLM, BertForSequenceClassification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as tud

from .args import ArgumentParserBuilder, OptionEnum, opt
from vizbert.model import ProjectionPursuitProbe
from vizbert.data import ConllWorkspace, ConllTextCollator, TrainingWorkspace, DATA_WORKSPACE_CLASSES,\
    ClassificationCollator, ZeroMeanTransform
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
                 OptionEnum.OUTPUT_FOLDER,
                 OptionEnum.USE_ZMT,
                 OptionEnum.OPTIMIZE_MEAN,
                 OptionEnum.DATASET,
                 opt('--project-type', type=str, choices=['pca', 'probe', 'tsne', 'svd'], default='probe'),
                 opt('--load-weights', type=str),
                 opt('--no-basic-tokenize', action='store_false', dest='basic_tokenize'),
                 opt('--limit', type=int, default=30))
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
        train_ds, dev_ds = dw.load_splits(splits=('train', 'dev',))
        collator = ClassificationCollator(tokenizer, multilabel=dev_ds.multilabel, max_length=128)
    train_loader = tud.DataLoader(train_ds, shuffle=True, batch_size=1, pin_memory=True, collate_fn=collator,
                                  num_workers=args.num_workers)
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

    try:
        args.output_folder.mkdir()
    except:
        pass

    probe = ProjectionPursuitProbe(768, args.probe_rank, optimize_mean=args.optimize_mean)
    if args.use_zmt:
        zmt = ZeroMeanTransform(768, 2).to(args.device)
        probe.zmt = zmt

    workspace.load_model(probe)
    proj_matrix = probe.orth_probe.to(args.device)
    use_pca = args.project_type == 'pca'
    use_svd = args.project_type == 'svd'
    use_tsne = args.project_type == 'tsne'
    probe.to(args.device)

    if use_pca or use_tsne or use_svd:
        states = []
        raw_texts = []
        colors = []
        with injector, torch.no_grad():
            for idx, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
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
            pca = PCA(args.probe_rank)
            pca.fit(states)
            torch.save(dict(b=torch.from_numpy(pca.components_)), 'pca.pt')
            lo_dim_model = pca
        elif use_svd:
            svd = TruncatedSVD(n_components=args.probe_rank)
            svd.fit(states)
            print(svd.explained_variance_ratio_, svd.singular_values_)
            torch.save(svd.components_, 'svd.pt')
            lo_dim_model = svd
        elif use_tsne:
            tsne = TSNE()
            states = tsne.fit_transform(states)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(states[:, 0], states[:, 1], c=colors, s=2)
            for pair, token in zip(states, raw_texts):  #('[CLS] ' + batch.raw_text[0] + ' [SEP]').split(' ')):
                ax.annotate(token, pair, size='xx-small')
            fig.savefig(args.output_folder / 'output-tsne.png', dpi=300)
            return

    with injector, torch.no_grad():
        coeffs_lst = []
        tokens_lst = []
        for idx, batch in enumerate(dev_loader):
            scores = model(batch.token_ids.to(args.device), attention_mask=batch.attention_mask.to(args.device))
            # scores = scores[0].softmax(1)[0]
            # colors = [(scores[0].item(), scores[1].item(), 0.0)] * batch.token_ids.size(1)
            data = reporting_module.buffer
            # coloring = compute_coloring(tokenizer, batch.raw_text[0])
            # assert len(coloring) + 2 == batch.token_ids.size(1)
            # data = merge_by_segmentation(data[0], coloring).unsqueeze(0)
            if use_pca or use_svd:
                coeffs = lo_dim_model.transform(data[0].cpu().numpy()).T
            else:
                if args.optimize_mean:
                    data = data - probe.mean.unsqueeze(0).unsqueeze(0).to(data.device)
                if args.use_zmt:
                    data = probe.zmt(data)
                coeffs = batch_gs_coeffs(proj_matrix, data).squeeze().cpu().numpy().T  # sequence, coeff
            coeffs_lst.append(coeffs)
            tokens_lst.extend([tokenizer.convert_ids_to_tokens(x) for x in batch.token_ids[0].tolist()])
            if idx == args.limit - 1:
                break
    if args.probe_rank > 1:
        coeffs = np.concatenate(coeffs_lst, 1)
    else:
        args.probe_rank = 2
        coeffs = np.concatenate(coeffs_lst, 0)
        coeffs = np.vstack((coeffs, np.random.uniform(-1, 1, coeffs.shape[0])))
    for i in range(args.probe_rank - 1):
        for j in range(i + 1, args.probe_rank):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(coeffs[i, :], coeffs[j, :], c='b', s=1.25)
            for pair, token in zip(coeffs[(i, j), :].T, tokens_lst):  # ('[CLS] ' + batch.raw_text[0] + ' [SEP]').split(' ')):
                ax.annotate(token, pair, size='xx-small')
            fig.savefig(args.output_folder / f'output-pp-{i}-{j}.png', dpi=300)


if __name__ == '__main__':
    main()
