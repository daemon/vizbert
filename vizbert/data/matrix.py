from scipy.sparse import vstack


def sparse_vstack(sparse_matrices):
    num_cols = max(m.shape[1] for m in sparse_matrices)
    for matrix in sparse_matrices:
        matrix.resize((matrix.shape[0], num_cols))
    return vstack(sparse_matrices)
