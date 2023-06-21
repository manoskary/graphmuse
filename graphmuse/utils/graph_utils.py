from scipy import sparse as sp
from scipy.sparse.linalg import eigs
import torch
import numpy as np


def degree(edge_index, num_nodes):
    """Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Parameters
    ----------
    index (LongTensor)
        Index tensor.

    num_nodes : (int)
        The number of nodes of the graph.
    """
    out = torch.zeros((num_nodes,), dtype=torch.long)
    one = torch.ones((edge_index[1].size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, edge_index[1], one)


def positional_encoding(edge_index, num_nodes, pos_enc_dim: int) -> torch.Tensor:
    """
    Graph positional encoding v/ Laplacian eigenvectors

    Parameters
    ----------
    edge_index (LongTensor)
        edge indices tensor.
    num_nodes : (int)
        The number of nodes of the graph.
    pos_enc_dim : int
        The Positional Encoding Dimension to be added to Nodes of the graph

    Returns
    -------
    pos_enc : torch.Tensor
        A tensor of shape (N, pos_enc_dim) where N is the number of nodes in the graph
    """
    # Laplacian
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)).to_dense()
    in_degree = A.sum(axis=1).numpy()
    N = sp.diags(in_degree.clip(1) ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with scipy
    EigVal, EigVec = eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    # increasing order
    EigVec = EigVec[:, EigVal.argsort()]
    pos_enc = torch.from_numpy(np.real(EigVec[:, 1:pos_enc_dim+1])).float()
    return pos_enc
