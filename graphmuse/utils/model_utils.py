import torch
import torch.nn as nn


class HeteroAttention(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super(HeteroAttention, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=0)


class HeteroConv(nn.Module):
    """
    Convert a Graph Convolutional module to a hetero GraphConv module.

    Parameters
    ----------
    module: torch.nn.Module
        Module to convert

    Returns
    -------
    module: torch.nn.Module
        Converted module
    """

    def __init__(self, in_features, out_features, etypes, in_edge_features=None, module:nn.Module=SageConv, bias=True, reduction='mean'):
        super(HeteroConv, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = module(in_features, out_features, bias=bias, in_edge_features=in_edge_features)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_features=None):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask], edge_features[mask, :] if edge_features is not None else None)
        return self.reduction(out).to(x.device)