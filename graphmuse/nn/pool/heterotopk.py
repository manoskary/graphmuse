from typing import Optional, Tuple, Dict, List
import torch
from torch import Tensor
from torch_geometric.nn.pool.connect import Connect
from torch_geometric.nn.pool.select import SelectOutput, SelectTopK
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
import torch_geometric.typing


class HeteroConnectOutput:
    r"""The output of the :class:`Connect` method, which holds the coarsened
    graph structure, and optional pooled edge features and batch vectors.

    Args:
        edge_index (torch.Tensor): The edge indices of the cooarsened graph.
        edge_attr (torch.Tensor, optional): The pooled edge features of the
            coarsened graph. (default: :obj:`None`)
        batch (torch.Tensor, optional): The pooled batch vector of the
            coarsened graph. (default: :obj:`None`)
    """
    edge_index_dict: Dict[str, Tensor]
    edge_attr_dict: Optional[Dict[str, Tensor]] = None
    batch_dict: Optional[Dict[str, Tensor]] = None

    def __init__(
        self,
        edge_index_dict: Dict[str, Tensor],
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
        batch_dict: Optional[Dict[str, Tensor]] = None
    ):
        if any([edge_index.dim() != 2 for edge_index in edge_index_dict.values()]):
            raise ValueError(f"Expected 'edge_index' to be two-dimensional ")

        if any([edge_index.dim() != 2 for edge_index in edge_index_dict.values()]):
            raise ValueError(f"Expected 'edge_index' to have size '2' in the ")

        if edge_attr_dict is not None and any([edge_attr.size(0) != edge_index.size(1) for edge_index, edge_attr in zip(edge_index_dict.values(), edge_attr_dict.values())]):
            raise ValueError(f"Expected 'edge_index' and 'edge_attr' to "
                             f"hold the same number of edges ")

        self.edge_index_dict = edge_index_dict
        self.edge_attr_dict = edge_attr_dict
        self.batch_dict = batch_dict


if torch_geometric.typing.WITH_PT113:
    HeteroConnectOutput = torch.jit.script(HeteroConnectOutput)


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
        batch_dict: Optional[Dict[str, Tensor]] = None,
    ) -> HeteroConnectOutput:

        if (not torch.jit.is_scripting() and all(
                [so.num_clusters!= so.cluster_index.size(0) for so in select_output.values()])):
            raise ValueError(f"'{self.__class__.__name__}' requires each "
                             f"cluster to contain only one node")

        for key in edge_index_dict.keys():
            src_key, _, dst_key = key
            if src_key == dst_key:
                edge_index_dict[key], edge_attr_dict[key] = filter_adj(
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
            batch_dict[node_key] = self.get_pooled_batch(so, batch_dict[node_key])

        return HeteroConnectOutput(edge_index_dict, edge_attr_dict, batch_dict)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


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
        self.node_types = node_types
        self.select = {k: SelectTopK(in_channels, ratio, min_score, nonlinearity) for k in node_types}
        self.connect = HeteroFilterEdges()
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for dt in self.node_types:
            self.select[dt].reset_parameters()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Tensor],
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
        batch_dict: Optional[Dict[str, Tensor]] = None,
        attn_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Optional[Dict[str, Tensor]], Optional[Dict[str, Tensor]], Dict[str, Tensor], Dict[str, Tensor]]:
        r"""Forward pass.

        Args:
            x_dict (torch.Tensor): The node feature matrix.
            edge_index_dict (torch.Tensor): The edge indices.
            edge_attr_dict (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch_dict (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn_dict (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch_dict is None:
            batch_dict = {k: torch.zeros(x_dict[k].size(0), dtype=torch.long, device=x_dict[k].device) for k in x_dict.keys()}

        attn_dict = x_dict if attn_dict is None else attn_dict
        select_out = {k: self.select[k](attn_dict[k], batch_dict[k]) for k in x_dict.keys()}

        perm = {k: select_out.node_index for k, select_out in select_out.items()}
        score = {k: select_out.weight for k, select_out in select_out.items()}
        assert score is not None

        # x_dict = x_dict[perm] * score.view(-1, 1)
        x_dict = {k: x_dict[k][perm[k]] * score[k].view(-1, 1) for k in x_dict.keys()}
        # x_dict = self.multiplier * x_dict if self.multiplier != 1 else x_dict
        x_dict = {k: self.multiplier * x_dict[k] if self.multiplier != 1 else x_dict[k] for k in x_dict.keys()}

        connect_out = self.connect(select_out, edge_index_dict, edge_attr_dict, batch_dict)
        return (x_dict, connect_out.edge_index_dict, connect_out.edge_attr_dict, connect_out.batch_dict, perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
