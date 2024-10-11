import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmuse.nn.models.metrical_gnn import MetricalGNN
import torch_scatter


class CadenceGNN(nn.Module):
    def __init__(self, metadata, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5, hybrid=False):
        super(CadenceGNN, self).__init__()
        self.gnn = MetricalGNN(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim // 2,
            num_layers=num_layers, metadata=metadata, dropout=dropout)

        hidden_dim = hidden_dim // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.hybrid = hybrid
        if self.hybrid:
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True,
                dropout=dropout)
            self.rnn_norm = nn.LayerNorm(hidden_dim)
            self.rnn_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.cat_proj = nn.Linear(hidden_dim*2, hidden_dim)
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cad_clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def hybrid_forward(self, x, batch):
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
        x, _ = self.rnn(x)
        x = self.rnn_norm(x)
        x = self.rnn_mlp(x)
        x = nn.utils.rnn.unpad_sequence(x, batch_first=True, lengths=lengths)
        x = torch.cat(x, dim=0)
        return x

    def encode(self, x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge, batch_note=None, onset_div=None):
        if batch_note is None:
            batch_note = torch.zeros((x_dict["note"].shape[0], ), device=x_dict["note"].device).long()
        if onset_div is None:
            # check if onset_div is present in x_dict
            if "onset_div" in x_dict:
                onset_div = x_dict.pop("onset_div")
        x = self.gnn(
            x_dict, edge_index_dict, neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge)
        if self.hybrid:
            z = self.hybrid_forward(x_dict["note"][neighbor_mask_node["note"] == 0], batch_note[neighbor_mask_node["note"] == 0])
            x = self.cat_proj(torch.cat((x, z), dim=-1))
        # repr_pred = self.repr_mlp(x)
        if onset_div is not None:
            # This is a pooling operation that aggregates the features of notes with the same onset
            batch_note = batch_note[neighbor_mask_node["note"] == 0]
            onset_div = onset_div[neighbor_mask_node["note"] == 0]
            a = torch.stack((batch_note, onset_div), dim=-1)
            unique, cluster = torch.unique(a, return_inverse=True, dim=0, sorted=True)
            multiplicity = torch.ones_like(batch_note)
            # mean the features of notes with the same onset
            x = torch.zeros(unique.size(0), x.size(1), device=x.device).scatter_add(0, cluster.unsqueeze(1).expand(-1, x.size(1)), x)
            multiplicity = torch.zeros(unique.size(0), device=x.device, dtype=torch.long).scatter_add(0, cluster, multiplicity)
            x = x / multiplicity.unsqueeze(1)
            # repr_out = self.repr_mlp(x)
            x = self.norm(x)
            x = self.pool_mlp(x)
            x = x[cluster]
            # repr_loss = F.mse_loss(repr_out, repr_pred)
        else:
        # NOTE: Remove above lines and replace with the following:
            onset_index = edge_index_dict["note", "onset", "note"]
            x = torch_scatter.scatter_mean(x[onset_index[0]], onset_index[1], dim=0, out=x.clone())
            x = self.norm(x)
            x = self.pool_mlp(x)
        return x

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None):
        if neighbor_mask_node is None:
            neighbor_mask_node = {k: torch.zeros((x_dict[k].shape[0], ), device=x_dict[k].device).long() for k in x_dict}
        if neighbor_mask_edge is None:
            neighbor_mask_edge = {k: torch.zeros((edge_index_dict[k].shape[-1], ), device=edge_index_dict[k].device).long() for k in edge_index_dict}
        x = self.encode(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        logits = self.cad_clf(x)
        return torch.softmax(logits, dim=-1)

    def clf(self, x):
        return self.cad_clf(x)
