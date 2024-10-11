import torch
import torch.nn as nn
from graphmuse.nn.models.metrical_gnn import MetricalGNN
import torch_geometric.nn as gnn
import numpy as np


class KeySignatureEncoder(object):
    """
    A class used to encode and decode key signature information from musical notes.

    Attributes
    ----------
    KEY_SIGNATURES : list
        A list of key signatures ranging from -7 to 7.
    encode_dim : int
        The dimension of the encoding space.
    classes_ : numpy.ndarray
        An array of unique key signature values.

    Methods
    -------
    encode(note_array)
        Encodes the key signature information from a note array.
    decode(x)
        Decodes the encoded key signature values back to their original form.
    """

    def __init__(self):
        self.KEY_SIGNATURES = list(range(-7, 8))
        self.encode_dim = len(self.KEY_SIGNATURES)
        self.classes_ = np.unique(self.KEY_SIGNATURES)

    def encode(self, note_array):
        """
        Encodes the key signature information from a note array.

        Parameters
        ----------
        note_array : numpy.ndarray
            An array containing note information, including key signature.

        Returns
        -------
        numpy.ndarray
            An array of encoded key signature values.
        """
        ks_array = note_array["ks_fifths"]
        return np.searchsorted(self.classes_, ks_array)

    def decode(self, x):
        """
        Decodes the encoded key signature values back to their original form.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            An array or tensor of encoded key signature values.

        Returns
        -------
        numpy.ndarray
            An array of decoded key signature values.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.classes_[x]


class PitchEncoder(object):
    """
    A class used to encode and decode musical pitch information.

    Attributes
    ----------
    PITCHES : dict
        A dictionary mapping pitch classes to their possible spellings.
    accepted_pitches : numpy.ndarray
        An array of all accepted pitch spellings.
    KEY_SIGNATURES : list
        A list of key signatures ranging from -7 to 7.
    encode_dim : int
        The dimension of the encoding space.
    classes_ : numpy.ndarray
        An array of unique pitch spellings.

    Methods
    -------
    rooting_function(x)
        Converts a pitch spelling triplet to a string representation.
    encode(note_array)
        Encodes pitch spelling triplets from a note array.
    decode(x)
        Decodes encoded pitch values back to their original form.
    """

    def __init__(self):
        self.PITCHES = {
            0: ["C", "B#", "D--"],
            1: ["C#", "B##", "D-"],
            2: ["D", "C##", "E--"],
            3: ["D#", "E-", "F--"],
            4: ["E", "D##", "F-"],
            5: ["F", "E#", "G--"],
            6: ["F#", "E##", "G-"],
            7: ["G", "F##", "A--"],
            8: ["G#", "A-"],
            9: ["A", "G##", "B--"],
            10: ["A#", "B-", "C--"],
            11: ["B", "A##", "C-"],
        }
        self.accepted_pitches = np.array([ii for i in self.PITCHES.values() for ii in i])
        self.KEY_SIGNATURES = list(range(-7, 8))
        self.encode_dim = len(self.accepted_pitches)
        self.classes_ = np.unique(self.accepted_pitches)

    def rooting_function(self, x):
        if x[1] == 0:
            suffix = ""
        elif x[1] == 1:
            suffix = "#"
        elif x[1] == 2:
            suffix = "##"
        elif x[1] == -1:
            suffix = "-"
        elif x[1] == -2:
            suffix = "--"
        out = x[0] + suffix
        return out

    def encode(self, note_array):
        """
        One-hot encoding of pitch spelling triplets.

        Parameters
        ----------
        note_array : numpy.ndarray
            An array containing note information, including pitch spelling.

        Returns
        -------
        numpy.ndarray
            An array of encoded pitch spelling values.
        """
        pitch_spelling = note_array[["step", "alter"]]
        root = self.rooting_function
        y = np.vectorize(root)(pitch_spelling)
        return np.searchsorted(self.classes_, y)

    def decode(self, x):
        """
        Decodes the encoded pitch values back to their original form.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            An array or tensor of encoded pitch values.

        Returns
        -------
        numpy.ndarray
            An array of decoded pitch values.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.classes_[x]


class PitchSpellingGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats_enc, n_layers, metadata, dropout=0.5, add_seq=False):
        super(PitchSpellingGNN, self).__init__()
        self.gnn_enc = MetricalGNN(in_feats, n_hidden, out_feats_enc, n_layers, metadata, dropout=dropout)
        self.normalize = gnn.GraphNorm(out_feats_enc)
        self.add_seq = add_seq
        self.pitch_label_encoder = PitchEncoder()
        self.key_label_encoder = KeySignatureEncoder()
        out_feats_pc = self.pitch_label_encoder.encode_dim
        out_feats_ks = self.key_label_encoder.encode_dim
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
