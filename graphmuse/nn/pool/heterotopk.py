from typing import Optional, Tuple, Dict, List
import torch
from torch import Tensor
from torch_geometric.nn.pool.connect import Connect, ConnectOutput
from torch_geometric.nn.pool.select import SelectOutput
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.pool.connect.filter_edges import filter_adj


def filter_hetero_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    node_index_src: Tensor,
    node_index_dst: Tensor,
    cluster_index_src: Optional[Tensor] = None,
    cluster_index_dst: Optional[Tensor] = None,
    num_nodes_src: Optional[int] = None,
    num_nodes_dst: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

    num_nodes_src = maybe_num_nodes(edge_index[0], num_nodes_src)
    num_nodes_dst = maybe_num_nodes(edge_index[1], num_nodes_dst)
    if cluster_index_src is None:
        cluster_index_src = torch.arange(node_index_src.size(0),
                                     device=node_index_src.device)
    if cluster_index_dst is None:
        cluster_index_dst = torch.arange(node_index_dst.size(0),
                                     device=node_index_dst.device)

    mask_src = node_index_src.new_full((num_nodes_src, ), -1)
    mask_src[node_index_src] = cluster_index_src

    mask_dst = node_index_dst.new_full((num_nodes_dst, ), -1)
    mask_dst[node_index_dst] = cluster_index_dst

    row, col = edge_index[0], edge_index[1]
    row, col = mask_src[row], mask_dst[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class HeteroFilterEdges(Connect):
    r"""Filters out edges if their incident nodes are not in any cluster.

    .. math::
            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where :math:`\mathbf{i}` denotes the set of retained nodes.
    It is assumed that each cluster contains only one node.
    """
    def forward(
        self,
        select_output: Dict[str, SelectOutput],
        edge_index_dict: Dict[str, Tensor],
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
        batch: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, ConnectOutput]:

        if (not torch.jit.is_scripting() and select_output.num_clusters
                != select_output.cluster_index.size(0)):
            raise ValueError(f"'{self.__class__.__name__}' requires each "
                             f"cluster to contain only one node")

        for key in edge_index_dict.keys():
            src_key, _, dst_key = key
            if src_key == dst_key:
                edge_attr_dict[key], edge_attr_dict[key] = filter_adj(
                    edge_index_dict[key],
                    edge_attr_dict[key],
                    select_output[src_key].node_index,
                    select_output[src_key].cluster_index,
                    num_nodes=select_output[src_key].num_nodes,
                )
            else:
                edge_index_dict[key], edge_attr_dict[key] = filter_hetero_adj(
                    edge_index_dict[key],
                    edge_attr_dict[key],
                    node_index_src=select_output[src_key].node_index,
                    node_index_dst=select_output[dst_key].node_index,
                    cluster_index_src=select_output[src_key].cluster_index,
                    cluster_index_dst=select_output[dst_key].cluster_index,
                    num_nodes_src=select_output[src_key].num_nodes,
                    num_nodes_dst=select_output[dst_key].num_nodes,
                )
        for node_key, so in select_output.items():
            batch = self.get_pooled_batch(so, batch[node_key])

        return {k: ConnectOutput(edge_index_dict[k], edge_attr_dict[k], batch[k[-1]]) for k in edge_index_dict.keys()}


class HeteroTopKPooling(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        node_types: List[str],
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: str = 'tanh',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.select = SelectTopK(in_channels, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.select.reset_parameters()

    def forward(
        self,
        x: Dict[Tensor],
        edge_index: Dict[Tensor],
        edge_attr: Optional[Dict[Tensor]] = None,
        batch: Optional[Dict[Tensor]] = None,
        attn: Optional[Dict[Tensor]] = None,
    ) -> Tuple[Dict[Tensor], Dict[Tensor], Optional[Dict[Tensor]], Optional[Dict[Tensor]], Dict[Tensor], Dict[Tensor]]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        x = x[perm] * score.view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (x, connect_out.edge_index, connect_out.edge_attr,
                connect_out.batch, perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')