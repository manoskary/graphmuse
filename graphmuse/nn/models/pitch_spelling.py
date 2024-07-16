import torch
import torch.nn as nn
from graphmuse.nn.models.metrical_gnn import MetricalGNN
import torch_geometric.nn as gnn


class PitchSpellingGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats_enc, out_feats_pc, out_feats_ks, n_layers, metadata, dropout=0.5, add_seq=False):
        super(PitchSpellingGNN, self).__init__()
        self.gnn_enc = MetricalGNN(in_feats, n_hidden, out_feats_enc, n_layers, metadata, dropout=dropout)
        self.normalize = gnn.GraphNorm(out_feats_enc)
        self.add_seq = add_seq
        if add_seq:
            self.rnn = nn.GRU(
                input_size=in_feats,
                hidden_size=n_hidden // 2,
                bidirectional=True,
                num_layers=1,
                batch_first=True,
            )
            self.rnn_norm = nn.LayerNorm(n_hidden)
            self.rnn_project = nn.Linear(n_hidden, out_feats_enc)
            self.cat_lin = nn.Linear(out_feats_enc * 2, out_feats_enc)
            self.rnn_ks = nn.GRU(
                input_size=out_feats_enc + out_feats_pc,
                hidden_size=n_hidden // 2,
                bidirectional=True,
                num_layers=1,
                batch_first=True,
            )
            self.rnn_norm_ks = nn.LayerNorm(n_hidden)
            self.rnn_project_ks = nn.Linear(n_hidden, out_feats_enc + out_feats_pc)


        self.mlp_clf_pc = nn.Sequential(
            nn.Linear(out_feats_enc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_pc),
        )
        self.mlp_clf_ks = nn.Sequential(
            nn.Linear(out_feats_enc + out_feats_pc, out_feats_enc // 2),
            nn.ReLU(),
            nn.LayerNorm(out_feats_enc // 2),
            nn.Dropout(dropout),
            nn.Linear(out_feats_enc // 2, out_feats_ks),
        )

    def sequential_forward(self, note, neighbor_mask_node, batch):
        z = note[neighbor_mask_node["note"] == 0]
        lengths = torch.bincount(batch)
        z = z.split(lengths.tolist())
        z = nn.utils.rnn.pad_sequence(z, batch_first=True)
        z, _ = self.rnn(z)
        z = self.rnn_norm(z)
        z = self.rnn_project(z)
        z = nn.utils.rnn.unpad_sequence(z, lengths, batch_first=True)
        z = torch.cat(z, dim=0)
        return z

    def sequential_ks_forward(self, x, batch):
        lengths = torch.bincount(batch)
        x = x.split(lengths.tolist())
        x = nn.utils.rnn.pad_sequence(x, batch_first=True)
        x, _ = self.rnn_ks(x)
        x = self.rnn_norm_ks(x)
        x = self.rnn_project_ks(x)
        x = nn.utils.rnn.unpad_sequence(x, lengths, batch_first=True)
        x = torch.cat(x, dim=0)
        return x

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None, batch=None):
        x = self.gnn_enc(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        x = self.normalize(x, batch=batch)
        if self.add_seq:
            z = self.sequential_forward(x_dict["note"], neighbor_mask_node, batch)
            x = torch.cat([x, z], dim=-1)
            x = self.cat_lin(x)
            out_pc = self.mlp_clf_pc(x)
            x = torch.cat([x, out_pc], dim=-1)
            x = self.sequential_ks_forward(x, batch)
            out_ks = self.mlp_clf_ks(x)
            return out_pc, out_ks

        out_pc = self.mlp_clf_pc(x)
        x = torch.cat([x, out_pc], dim=-1)
        out_ks = self.mlp_clf_ks(x)
        return out_pc, out_ks
