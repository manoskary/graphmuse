import torch.nn as nn
import torch
import torch.nn.functional as F


class JumpingKnowledge(nn.Module):
    """
    Combines information per GNN layer with a LSTM,
    provided that all hidden representation are on the same dimension.
    """
    def __init__(self, n_hidden, n_layers):
        super(JumpingKnowledge, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, xs):
        x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=1)


class JumpingKnowledge3D(nn.Module):
    """
    Combines information per GNN layer with a LSTM,
    provided that all hidden representation are on the same dimension.
    """
    def __init__(self, n_hidden, n_layers):
        super(JumpingKnowledge3D, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.n_hidden = n_hidden
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, xs):
        x = torch.stack(xs, dim=2)  # [batch_size, num_nodes, num_layers, num_channels]
        h = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        alpha, _ = self.lstm(h)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1).view(x.shape[0], x.shape[1], self.n_hidden)


