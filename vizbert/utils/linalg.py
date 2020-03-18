import random

from scipy.linalg import orth, norm
from scipy.sparse import vstack
import numpy as np
import torch


__all__ = ['sparse_vstack', 'gramschmidt_project', 'sample_subspace_noise', 'orth_compl', 'batch_gs_project']


def sparse_vstack(sparse_matrices):
    num_cols = max(m.shape[1] for m in sparse_matrices)
    for matrix in sparse_matrices:
        matrix.resize((matrix.shape[0], num_cols))
    return vstack(sparse_matrices)


def gramschmidt_project(q, x, strength=1):
    return x - strength * (np.dot(x, q) / np.dot(q, q)) * q


def batch_gs_project(q: torch.Tensor, x: torch.Tensor, strength=1):
    v = torch.einsum('i,bsi->bs', q, x)
    v = v / q.dot(q)
    q = q.unsqueeze(0).unsqueeze(0).expand_as(x)
    v = v.unsqueeze(-1).expand_as(q)
    return x - strength * v * q


def orth_compl(A: np.ndarray):
    Q = orth(A)
    compl_rank = Q.shape[0] - Q.shape[1]
    qvecs = list(iter(Q.T))
    new_vecs = []
    for _ in range(compl_rank):
        new_vec = np.random.uniform(-1, 1, Q.shape[0])
        for qvec in qvecs:
            new_vec = gramschmidt_project(qvec, new_vec)
            new_vec = new_vec / np.linalg.norm(new_vec)
        qvecs.append(new_vec)
        new_vecs.append(new_vec)
    Qc = np.stack(new_vecs).T.astype(np.float32)
    return Q, Qc


def sample_subspace_noise(Q: np.ndarray, a: float, b: float, length: int = 1):
    vecs = []
    for _ in range(length):
        vec = np.zeros(Q.shape[0], dtype=np.float32)
        for q in Q.T:
            vec += (a + random.random() * (b - a)) * q.astype(np.float32)
        vec = (b - a) * vec / norm(vec)
        vecs.append(vec)
    q_noise = np.stack(vecs)
    if q_noise.shape[0] == 1:
        q_noise = q_noise.squeeze(0)
    return q_noise
