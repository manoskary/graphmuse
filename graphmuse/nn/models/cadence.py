import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmuse.nn.models.metrical_gnn import MetricalGNN
import torch_scatter


class CadenceGNN(nn.Module):
    """
    CadenceGNN is a graph neural network model for cadence detection in music.

    Parameters
    ----------
    metadata : dict
        Metadata for the graph.
    input_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    output_channels : int
        Number of output channels.
    num_layers : int
        Number of layers in the GNN.
    dropout : float, optional
        Dropout rate, by default 0.5.
    hybrid : bool, optional
        Whether to use a hybrid model with RNN, by default False.

    Examples
    --------
    >>> metadata = {"note": {"x": torch.randn(10, 16)}}
    >>> model = CadenceGNN(metadata, input_channels=16, hidden_channels=32, output_channels=2, num_layers=3)
    >>> x_dict = {"note": torch.randn(10, 16)}
    >>> edge_index_dict = {"note": {"note": torch.randint(0, 10, (2, 20))}}
    >>> out = model(x_dict, edge_index_dict)
    >>> print(out.shape)
    torch.Size([10, 2])
    """
    def __init__(self, metadata, input_channels, hidden_channels, output_channels, num_layers, dropout=0.5, hybrid=False):
        super(CadenceGNN, self).__init__()
        self.gnn = MetricalGNN(
            input_channels=input_channels, hidden_channels=hidden_channels, output_channels=hidden_channels // 2,
            num_layers=num_layers, metadata=metadata, dropout=dropout)

        hidden_channels = hidden_channels // 2
        self.norm = nn.LayerNorm(hidden_channels)
        self.hybrid = hybrid
        if self.hybrid:
            self.rnn = nn.GRU(
                input_size=input_channels, hidden_size=hidden_channels//2, num_layers=2, batch_first=True, bidirectional=True,
                dropout=dropout)
            self.rnn_norm = nn.LayerNorm(hidden_channels)
            self.rnn_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(hidden_channels),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.cat_proj = nn.Linear(hidden_channels*2, hidden_channels)
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cad_clf = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, output_channels),
        )

    def hybrid_forward(self, x, batch):
        """
        Forward pass for the hybrid model with RNN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        batch : torch.Tensor
            Batch indices.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
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
        """
        Encode the input graph data.

        Parameters
        ----------
        x_dict : dict
            Dictionary of input node features.
        edge_index_dict : dict
            Dictionary of edge indices.
        neighbor_mask_node : dict
            Dictionary of neighbor masks for nodes.
        neighbor_mask_edge : dict
            Dictionary of neighbor masks for edges.
        batch_note : torch.Tensor, optional
            Batch indices for notes, by default None.
        onset_div : torch.Tensor, optional
            Onset divisions, by default None.

        Returns
        -------
        torch.Tensor
            Encoded node features.
        """
        if batch_note is None:
            batch_note = torch.zeros((x_dict["note"].shape[0], ), device=x_dict["note"].device).long()
        if onset_div is None:
            if "onset_div" in x_dict:
                onset_div = x_dict.pop("onset_div")
        x = self.gnn(
            x_dict, edge_index_dict, neighbor_mask_node=neighbor_mask_node, neighbor_mask_edge=neighbor_mask_edge)
        if self.hybrid:
            z = self.hybrid_forward(x_dict["note"][neighbor_mask_node["note"] == 0], batch_note[neighbor_mask_node["note"] == 0])
            x = self.cat_proj(torch.cat((x, z), dim=-1))
        if onset_div is not None:
            batch_note = batch_note[neighbor_mask_node["note"] == 0]
            onset_div = onset_div[neighbor_mask_node["note"] == 0]
            a = torch.stack((batch_note, onset_div), dim=-1)
            unique, cluster = torch.unique(a, return_inverse=True, dim=0, sorted=True)
            multiplicity = torch.ones_like(batch_note)
            x = torch.zeros(unique.size(0), x.size(1), device=x.device).scatter_add(0, cluster.unsqueeze(1).expand(-1, x.size(1)), x)
            multiplicity = torch.zeros(unique.size(0), device=x.device, dtype=torch.long).scatter_add(0, cluster, multiplicity)
            x = x / multiplicity.unsqueeze(1)
            x = self.norm(x)
            x = self.pool_mlp(x)
            x = x[cluster]
        else:
            onset_index = edge_index_dict["note", "onset", "note"]
            x = torch_scatter.scatter_mean(x[onset_index[0]], onset_index[1], dim=0, out=x.clone())
            x = self.norm(x)
            x = self.pool_mlp(x)
        return x

    def forward(self, x_dict, edge_index_dict, neighbor_mask_node=None, neighbor_mask_edge=None):
        """
        Forward pass for the CadenceGNN model.

        Parameters
        ----------
        x_dict : dict
            Dictionary of input node features.
        edge_index_dict : dict
            Dictionary of edge indices.
        neighbor_mask_node : dict, optional
            Dictionary of neighbor masks for nodes, by default None.
        neighbor_mask_edge : dict, optional
            Dictionary of neighbor masks for edges, by default None.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        if neighbor_mask_node is None:
            neighbor_mask_node = {k: torch.zeros((x_dict[k].shape[0], ), device=x_dict[k].device).long() for k in x_dict}
        if neighbor_mask_edge is None:
            neighbor_mask_edge = {k: torch.zeros((edge_index_dict[k].shape[-1], ), device=edge_index_dict[k].device).long() for k in edge_index_dict}
        x = self.encode(x_dict, edge_index_dict, neighbor_mask_node, neighbor_mask_edge)
        logits = self.cad_clf(x)
        return torch.softmax(logits, dim=-1)

    def clf(self, x):
        """
        Classify the input node features.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Output node features.
        """
        return self.cad_clf(x)
