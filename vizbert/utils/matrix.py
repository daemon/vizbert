import random

from scipy.linalg import orth
from scipy.sparse import vstack
import numpy as np


def sparse_vstack(sparse_matrices):
    num_cols = max(m.shape[1] for m in sparse_matrices)
    for matrix in sparse_matrices:
        matrix.resize((matrix.shape[0], num_cols))
    return vstack(sparse_matrices)


def orth_compl(A: np.ndarray):
    Q = orth(A)
    compl_rank = Q.shape[0] - Q.shape[1]
    qvecs = list(iter(Q.T))
    new_vecs = []
    for _ in range(compl_rank):
        new_vec = np.random.uniform(-1, 1, Q.shape[0])
        for qvec in qvecs:
            new_vec = new_vec - (np.dot(new_vec, qvec) / np.dot(qvec, qvec)) * qvec
            new_vec = new_vec / np.linalg.norm(new_vec)
        qvecs.append(new_vec)
        new_vecs.append(new_vec)
    Qc = np.stack(new_vecs).T
    return Q, Qc


def sample_subspace_noise(Q: np.ndarray, a: float, b: float, length: int = 1):
    vecs = []
    for _ in range(length):
        vec = np.zeros(Q.shape[1])
        for q in Q.T:
            vec += (a + random.random() * (b - a)) * q
        vecs.append(vec)
    q_noise = np.stack(vecs)
    if q_noise.shape[0] == 1:
        q_noise = q_noise.squeeze(0)
    return q_noise
