import torch
import torch.nn as nn

class GraphConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer implementation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    dropout : float
        Dropout rate.
    bias : bool, optional
        Whether to include a bias term, by default True.

    Examples
    --------
    >>> gcn_layer = GraphConv(input_channels=16, output_channels=8, dropout=0.5)
    >>> h = torch.randn(10, 16)
    >>> adj = torch.randint(0, 2, (10, 10))
    >>> out = gcn_layer(h, adj)
    >>> print(out.shape)
    torch.Size([10, 8])
    """
    def __init__(self, input_channels, output_channels, dropout, bias=True):
        super(GraphConv, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dropout = dropout
        self.bias = bias

        self.W = nn.Parameter(torch.zeros(size=(input_channels, output_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(size=(output_channels,)))
            nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, adj):
        """
        Forward pass for the Graph Convolutional Network (GCN) layer.

        Parameters
        ----------
        h : torch.Tensor
            Input node features.
        adj : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        Wh = torch.mm(h, self.W)
        h_prime = torch.matmul(adj, Wh)
        if self.bias:
            return h_prime + self.b
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_channels) + ' -> ' \
            + str(self.output_channels) + ')'
