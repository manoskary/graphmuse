from torch_geometric.nn import SAGEConv as SageConvLayer
import torch.nn as nn
import torch
import torch.nn.functional as F


class GGRU(nn.Module):
    """
    A GRU inspired implementation of a GCN cell.

    h(t-1) in this case is the neighbors of node t.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    bias : bool, optional
        Whether to include a bias term, by default False.

    Examples
    --------
    >>> ggru_layer = GGRU(input_channels=16, output_channels=8, bias=True)
    >>> x = torch.randn(10, 16)
    >>> adj = torch.randint(0, 2, (10, 10))
    >>> out = ggru_layer(x, adj)
    >>> print(out.shape)
    torch.Size([10, 8])
    """
    def __init__(self, input_channels, output_channels, bias=False):
        super(GGRU, self).__init__()
        self.wr = SageConvLayer(input_channels, input_channels)
        self.wz = SageConvLayer(input_channels, output_channels)
        self.w_ni = nn.Linear(input_channels*2, output_channels)
        self.w_nh = nn.Linear(input_channels, input_channels)
        self.proj = nn.Linear(input_channels, output_channels)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the GGRU layer.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.w_ni.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_nh.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)

    def forward(self, x, adj):
        """
        Forward pass for the GGRU layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        h = x
        r = F.sigmoid(self.wr(h, adj))
        z = F.sigmoid(self.wz(h, adj))
        h = torch.bmm(adj, self.w_nh(h)) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        n = self.w_ni(torch.cat([x, r*h], dim=-1))
        n = F.tanh(n)
        neigh = torch.bmm(adj, x) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        out = (1 - z)*n + z*self.proj(neigh)
        return out
