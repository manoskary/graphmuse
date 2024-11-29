import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing, MessageNorm


class MusGConv(MessagePassing):
    """
    MusGConv is a message passing neural network layer for graph data.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    in_edge_channels : int, optional
        Number of input edge channels, by default 0.
    bias : bool, optional
        Whether to include a bias term, by default True.
    return_edge_emb : bool, optional
        Whether to return edge embeddings, by default False.
    norm_msg : bool, optional
        Whether to normalize messages, by default False.
    **kwargs
        Additional arguments for the MessagePassing class.

    Examples
    --------
    >>> import torch
    >>> from graphmuse.nn.conv.musgconv import MusGConv
    >>> x = torch.randn(10, 16)
    >>> edge_index = torch.randint(0, 10, (2, 20))
    >>> edge_attr = torch.randn(20, 16)
    >>> conv = MusGConv(input_channels=16, output_channels=32)
    >>> out = conv(x, edge_index, edge_attr)
    >>> print(out.shape)
    torch.Size([10, 32])
    """
    def __init__(self, input_channels, output_channels, in_edge_channels=0, bias=True, return_edge_emb=False, norm_msg=False, **kwargs):
        super().__init__(aggr='add')
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.return_edge_emb = return_edge_emb
        self.aggregation = kwargs.get("aggregation", "cat")
        self.in_edge_channels = in_edge_channels if in_edge_channels > 0 else input_channels
        self.lin = nn.Linear(input_channels, output_channels)
        self.msg_norm = MessageNorm() if norm_msg else nn.Identity()
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.in_edge_channels, output_channels),
            nn.ReLU(),
            nn.LayerNorm(output_channels),
            nn.Linear(output_channels, output_channels),
        )
        self.proj = nn.Linear(3 * output_channels, output_channels) if self.aggregation == "cat" else nn.Linear(2 * output_channels, output_channels)
        self.bias = nn.Parameter(torch.zeros(output_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the MusGConv layer.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[3].weight, gain=gain)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the MusGConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_attr : torch.Tensor
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        if edge_attr is None:
            edge_attr = torch.abs(x[edge_index[0]] - x[edge_index[1]])
        x = self.lin(x)
        edge_attr = self.edge_mlp(edge_attr)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h = self.proj(torch.cat((x, h), dim=-1))
        if self.bias is not None:
            h = h + self.bias
        if self.return_edge_emb:
            return h, edge_attr
        return h

    def message(self, x_j, edge_attr):
        """
        Compute messages for the MusGConv layer.

        Parameters
        ----------
        x_j : torch.Tensor
            Input node features of the neighbors.
        edge_attr : torch.Tensor
            Edge attributes.

        Returns
        -------
        torch.Tensor
            Messages.
        """
        if self.aggregation == "cat":
            msg = torch.cat((x_j, edge_attr), dim=-1)
        elif self.aggregation == "add":
            msg = x_j + edge_attr
        elif self.aggregation == "mul":
            msg = x_j * edge_attr
        else:
            raise ValueError("Aggregation type not supported")

        msg = self.msg_norm(x_j, msg)
        return msg
