import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import Tensor
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData, Batch
from graphmuse.samplers import random_score_region_torch, SubgraphMultiplicitySampler
from torch_geometric.sampler.utils import to_csc, to_hetero_csc, remap_keys
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, Dict
from torch_geometric.typing import EdgeType, NodeType, WITH_PYG_LIB
from torch_geometric.loader.utils import (
    filter_data,
    infer_filter_per_worker,
    filter_hetero_data
)
from torch_geometric.sampler.base import NumNeighbors


NumNeighborsType = Union[NumNeighbors, List[int], Dict[EdgeType, List[int]]]


class MuseNeighborLoader(DataLoader):
    """
    Dataloader for MuseData objects. It samples a random region of a given budget from the graph.
    If the budget is larger than the number of nodes, it returns all nodes.
    If the graph is heterogeneous, it samples from the given node type.
    """
    def __init__(
            self,
            graphs: List[Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]]],
            num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
            subgraph_size: int = 100,
            transform: Optional[Callable] = None,
            transform_sampler_output: Optional[Callable] = None,
            filter_per_worker: Optional[bool] = None,
            custom_cls: Optional[HeteroData] = None,
            device: Union[str, torch.device] = "cpu",
            is_sorted: bool = False,
            share_memory: bool = False,
            **kwargs,
        ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(graphs)

        self.graphs = graphs
        self.metadata = graphs[0].metadata()
        self.num_neighbors = num_neighbors
        self.subgraph_size = subgraph_size
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.device = device
        self.is_sorted = is_sorted
        self.share_memory = share_memory
        self.node_time: Optional[Dict[NodeType, Tensor]] = None
        self.edge_time: Optional[Dict[EdgeType, Tensor]] = None

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        self.batch_size = kwargs.pop('batch_size', 1)
        input_type = "note"
        base_sampler = SubgraphMultiplicitySampler(graphs, max_subgraph_size=subgraph_size, batch_size=self.batch_size,
                                                   multiplicity_ratio=2)
        # Get node type (or `None` for homogeneous graphs):

        dataset = ConcatDataset([self.graphs])
        super().__init__(dataset, collate_fn=self.collate_fn, batch_size=1, batch_sampler=base_sampler, **kwargs)

    def __call__(
            self,
            index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        return out

    def collate_fn(self, data_batch: Union[Tensor, List[int]]) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        data_list = []
        for data in data_batch:
            if data["note"].num_nodes <= self.subgraph_size:
                data_list.append(data)
                continue
            # sample nodes
            target_nodes = random_score_region_torch(data, self.subgraph_size, node_type="note")
            # Convert the graph data into CSC format for sampling:
            to_rel_type = {k: '__'.join(k) for k in data.edge_types}
            to_edge_type = {v: k for k, v in to_rel_type.items()}
            colptr_dict, row_dict, perm = to_hetero_csc(
                data, device='cpu', share_memory=self.share_memory,
                is_sorted=self.is_sorted)

            row_dict = remap_keys(row_dict, to_rel_type)
            colptr_dict = remap_keys(colptr_dict, to_rel_type)
            if WITH_PYG_LIB:
                args = (
                    data.node_types,
                    data.edge_types,
                    colptr_dict,
                    row_dict,
                    target_nodes, # seed_dict
                    self.num_neighbors.get_mapped_values(self.edge_types),
                    self.node_time,
                )
                args += (
                    True,  # csc
                    False, # do not replace
                    True, # Subgraph not induced
                    False, # not disjoint
                    'uniform', # temporal strategy
                    True,  # return_edge_id
                )

                out = torch.ops.pyg.hetero_neighbor_sample(*args)
                row, col, node, edge, batch = out[:4] + (None,)
            else:
                out = torch.ops.torch_sparse.hetero_neighbor_sample(
                        data.node_types,
                        data.edge_types,
                        colptr_dict,
                        row_dict,
                        target_nodes,  # seed_dict
                        self.num_neighbors.get_mapped_values(data.edge_types),
                        self.num_neighbors.num_hops,
                        True, # subgraph not induced
                        False, # do not replace
                    )
                node, row, col, edge, batch = out + (None,)


            # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
            num_sampled_nodes = num_sampled_edges = None
            if len(out) >= 6:
                num_sampled_nodes, num_sampled_edges = out[4:6]
            row = remap_keys(row, to_edge_type)
            col = remap_keys(col, to_edge_type)
            edge = remap_keys(edge, to_edge_type)
            # filter data to create a new HeteroData object.
            data_out = filter_hetero_data(data, node, row, col, edge, None)
            data_list.append(data_out)
        batch_out = Batch.from_data_list(data_list)
        return batch_out

    @property
    def num_neighbors(self) -> NumNeighbors:
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        if isinstance(num_neighbors, NumNeighbors):
            self._num_neighbors = num_neighbors
        else:
            self._num_neighbors = NumNeighbors(num_neighbors)

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'