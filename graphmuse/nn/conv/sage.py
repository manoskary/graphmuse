import torch.nn as nn
import torch


class SageConvLayer(nn.Module):
    """
    GraphSAGE Convolutional Layer.

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
    >>> layer = SageConvLayer(input_channels=16, output_channels=32)
    >>> features = torch.randn(10, 16)
    >>> adj = torch.randint(0, 2, (10, 10))
    >>> output = layer.forward_adj(features, adj)
    >>> print(output.shape)
    torch.Size([10, 32])
    """
    def __init__(self, input_channels, output_channels, bias=False):
        super(SageConvLayer, self).__init__()
        self.neigh_linear = nn.Linear(input_channels, input_channels, bias=bias)
        self.linear = nn.Linear(input_channels * 2, output_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the SageConvLayer.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)

    def forward_adj(self, features, adj, neigh_feats=None):
        """
        Forward pass for the SageConvLayer using adjacency matrix.

        Parameters
        ----------
        features : torch.Tensor
            The node features.
        adj : torch.LongTensor
            The edge indices size (N Neighbors, M Target).
        neigh_feats : torch.Tensor, optional
            The neighbor features, by default None.

        Returns
        -------
        torch.Tensor
            The output features.
        """
        if neigh_feats is None:
            neigh_feats = features
        h = self.neigh_linear(neigh_feats)
        if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                h = torch.bmm(adj, h) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                h = torch.mm(adj, h) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            h = torch.mm(adj, h) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)
        z = self.linear(torch.cat([features, h], dim=-1))
        return z
    #
    # def forward(self, features, edge_index):
    #     """
    #     This is the forward pass for the edge_index version of adjacency.
    #
    #     It is in the style of PYG.
    #
    #     Parameters
    #     ----------
    #     features : torch.Tensor
    #         The node features.
    #     edge_index : torch.LongTensor
    #         The edge indices size (2, num_edges).
    #     """
    #     h = self.neigh_linear(features)[edge_index[1]]
    #     s = scatter(scr=h, index=edge_index[0], dim=0, dim_size=features.size(0), reduce='mean')
    #     z = self.linear(torch.cat([features, s], dim=-1))
    #     return z
