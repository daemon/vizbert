from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
import numpy as np
import torch

from vizbert.utils.cca import get_cca_similarity
from .args import opt, ArgumentParserBuilder


def main():
    def find_svd_idx(expl_variance_ratio, threshold=0.99):
        idx = 0
        val = 0
        while val < threshold:
            val += expl_variance_ratio[idx]
            idx += 1
        return idx
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--rank', type=int, default=2),
                 opt('--input', '-i', type=str, required=True),
                 opt('--output', '-o', type=str, required=True),
                 opt('--linear-input', '-li', type=str),
                 opt('--method', '-m', type=str, default='svd', choices=('svd', 'lin')))
    args = apb.parser.parse_args()

    hidden_states, class_outputs = torch.load(args.input)
    class_outputs = [y.expand(x.size(1), -1) for x, y in zip(hidden_states, class_outputs)]
    X = torch.cat(hidden_states, 1)
    X = X.squeeze(0)
    y = torch.cat(class_outputs, 0)
    print(y.size())
    y = y.squeeze(0)
    print(X.size())
    del hidden_states
    del class_outputs
    if args.method == 'svd':
        svd = TruncatedSVD(args.rank, n_iter=10)
        svd.fit(X.cpu().numpy())
        print('sum var ratio', np.sum(svd.explained_variance_ratio_))
        torch.save((svd.components_, svd.explained_variance_ratio_), args.output)
    elif args.method == 'svcca':
        print('Fitting')
        x_svd = TruncatedSVD(min(*X.shape) - 1)
        x_svd.fit(X.cpu().numpy())
        print('Fitting y')
        y_svd = TruncatedSVD(min(*y.shape) - 1)
        y_svd.fit(y.cpu().numpy())
        x_dirs = x_svd.components_[:find_svd_idx(x_svd.explained_variance_ratio_)]
        y_dirs = y_svd.components_[:find_svd_idx(y_svd.explained_variance_ratio_)]
        ret = get_cca_similarity(y_svd, x_svd, compute_dirns=True)
        print(ret)
    elif args.method == 'lin':
        model = LinearRegression(fit_intercept=False, copy_X=False)
        with torch.no_grad():
            if args.linear_input:
                print('Projecting out old one...')
                U, _ = torch.load(args.linear_input)  # type: torch.Tensor
                U = U.T
                U = torch.from_numpy(U)
                P = torch.eye(max(*U.shape)) - U.matmul(U.t())
                print(P.shape)
                X = torch.einsum('ij,bj->bi', P.to(X.device), X)
            model.fit(X.cpu().numpy(), y.cpu().numpy())
            if args.linear_input:
                V = torch.svd(torch.cat((U.t(), torch.from_numpy(model.coef_))))[2].T.cpu().numpy()
                print(V.shape)
                torch.save((V, [0.0]), args.output)
            else:
                torch.save((torch.svd(torch.from_numpy(model.coef_))[2].cpu().numpy().T, [0.0]), args.output)


if __name__ == '__main__':
    main()
