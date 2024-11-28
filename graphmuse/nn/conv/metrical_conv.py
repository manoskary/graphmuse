from .sage import SageConvLayer
import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import SAGEConv


class MetricalConvLayer(nn.Module):
    """
    Metrical Convolutional Layer implementation.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    activation : callable, optional
        Activation function, by default None.
    dropout : float, optional
        Dropout rate, by default 0.2.
    bias : bool, optional
        Whether to include a bias term, by default True.

    Examples
    --------
    >>> mcl = MetricalConvLayer(input_channels=16, output_channels=8, activation=torch.nn.ReLU(), dropout=0.5)
    >>> x_metrical = torch.randn(10, 16)
    >>> x = torch.randn(10, 16)
    >>> edge_index = torch.randint(0, 10, (2, 20))
    >>> batch = torch.randint(0, 2, (10,))
    >>> out = mcl(x_metrical, x, edge_index, batch)
    >>> print(out.shape)
    torch.Size([10, 8])
    """
    def __init__(self, input_channels, output_channels, activation=None, dropout=0.2, bias=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation = nn.Identity() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.BatchNorm1d(output_channels)
        self.neigh = nn.Linear(input_channels, input_channels, bias=bias)
        self.conv_out = nn.Linear(3 * input_channels, output_channels, bias=bias)
        self.seq = SAGEConv(input_channels, input_channels, aggr='add')

    def reset_parameters(self):
        """
        Reset the parameters of the MetricalConvLayer.
        """
        self.neigh.reset_parameters()
        self.conv_out.reset_parameters()
        self.seq.reset_parameters()

    def forward(self, x_metrical, x, edge_index, batch):
        """
        Forward pass for the Metrical Convolutional Layer.

        Parameters
        ----------
        x_metrical : torch.Tensor
            Input metrical node features.
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.
        batch : torch.Tensor
            Batch indices.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        if batch is None or torch.all(batch == batch[0]):
            seq_index = torch.vstack((torch.arange(0, x_metrical.size(0) - 1), torch.arange(1, x_metrical.size(0)))).long()
        else:
            seq_index = []
            lengths = torch.unique(batch, return_counts=True)[1]
            lengths = torch.cat((torch.zeros(1, dtype=torch.long), lengths))
            lengths_cummulative = torch.cumsum(lengths, dim=0)
            for i in range(len(lengths) - 1):
                if lengths[i] == 1:
                    continue
                seq_index.append(torch.vstack((torch.arange(lengths_cummulative[i], lengths_cummulative[i + 1] - 1), torch.arange(lengths_cummulative[i] + 1, lengths_cummulative[i + 1]))))
            seq_index = torch.cat(seq_index, dim=1)
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()

        h_neigh = self.neigh(x)
        h_scatter = scatter_add(h_neigh[edge_index[0]], edge_index[1], dim=0, dim_size=x_metrical.size(0))
        h_seq = self.seq(x_metrical, seq_index.to(x_metrical.device))
        h = torch.cat([h_scatter, x_metrical, h_seq], dim=1)
        h = self.conv_out(h)
        h = self.activation(h)
        h = self.normalize(h)
        h = self.dropout(h)
        out = scatter_add(h[edge_index[1]], edge_index[0], dim=0, dim_size=x.size(0))
        return out
