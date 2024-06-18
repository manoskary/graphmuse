import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import Tensor
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData, Batch
from graphmuse.samplers import random_score_region_torch, SubgraphMultiplicitySampler
from torch_geometric.sampler.utils import to_csc, to_hetero_csc, remap_keys
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, Dict
from torch_geometric.typing import EdgeType, NodeType, WITH_PYG_LIB
import torch_geometric
from torch_geometric.data import InMemoryDataset
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
            graphs: Union[List[Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]]], InMemoryDataset],
            num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
            subgraph_size: int = 100,
            subgraph_sample_ratio: float = 2,
            transform: Optional[Callable] = None,
            transform_sampler_output: Optional[Callable] = None,
            filter_per_worker: Optional[bool] = None,
            custom_cls: Optional[HeteroData] = None,
            device: Union[str, torch.device] = "cpu",
            is_sorted: bool = False,
            share_memory: bool = False,
            order_batch: bool = True,
            **kwargs,
        ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(graphs)
        self.is_dataset = isinstance(graphs, InMemoryDataset)
        self.graphs = graphs
        self.nlengths = np.array([graphs[i].num_nodes for i in range(len(graphs))]) if self.is_dataset else np.array([g.num_nodes for g in graphs])
        self.metadata = graphs[0].metadata()
        self.num_neighbors = num_neighbors if num_neighbors is not None else {k: [0] for k in self.metadata[1]}
        self.subgraph_size = subgraph_size
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.device = device
        self.order_batch = order_batch
        self.is_sorted = is_sorted
        self.share_memory = share_memory
        self.node_time: Optional[Dict[NodeType, Tensor]] = None
        self.edge_time: Optional[Dict[EdgeType, Tensor]] = None
        self.edge_weight: Optional[Dict[EdgeType, Tensor]] = None
        # Fetch neighbors is set to False when the num neighbors is None
        self.fetch_neighbors = num_neighbors is not None

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        self.batch_size = kwargs.pop('batch_size', 1)
        input_type = "note"
        base_sampler = SubgraphMultiplicitySampler(self.nlengths, max_subgraph_size=subgraph_size, batch_size=self.batch_size,
                                                   multiplicity_ratio=subgraph_sample_ratio)
        # Get node type (or `None` for homogeneous graphs):

        dataset = ConcatDataset([self.graphs]) if not self.is_dataset else self.graphs
        super().__init__(dataset, collate_fn=self.collate_fn, batch_size=1, batch_sampler=base_sampler, **kwargs)

    def __call__(
            self,
            index: List[HeteroData],
    ) -> Union[Data, HeteroData]:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.collate_fn(index)
        return out

    def collate_fn(self, data_batch: List[HeteroData]) -> Batch:
        r"""Samples a subgraph from a batch of input nodes."""
        data_list = []
        target_nodes = []
        for data in data_batch:
            data = data.contiguous()
            data_out, target_out = self.sample_from_each_graph(data)
            data_list.append(data_out)
            target_nodes.append(target_out)
        # re-order the data list based on the number of target nodes in descending order
        if self.order_batch:
            target_nodes = np.array(target_nodes)
            idx = np.argsort(target_nodes)[::-1]
            data_list = [data_list[i] for i in idx]
        # create a batch object
        batch_out = Batch.from_data_list(data_list)
        if self.transform is not None:
            batch_out = self.transform(batch_out, self.num_neighbors.num_hops)
        return batch_out

    def sample_from_each_graph(self, data):
        # If the graph is already smaller than the subgraph size, return the whole graph
        if data["note"].num_nodes <= self.subgraph_size:
            # target_lenghts = {k: data[k].x.shape[0] for k in data.node_types}
            # for k, v in target_lenghts.items():
            #     data[k].num_sampled_nodes = v
            if WITH_PYG_LIB and self.fetch_neighbors:
                self.set_neighbor_mask_node(data, {k: [v.shape[0]] for k, v in data.x_dict.items()})
                self.set_neighbor_mask_edge(data, {k: [v.shape[1]] for k, v in data.edge_index_dict.items()})
            return data, data["note"].num_nodes
        # sample nodes
        target_nodes = random_score_region_torch(data, self.subgraph_size, node_type="note")
        target_lenghts = {k: v.shape[0] for k, v in target_nodes.items()}
        # Convert the graph data into CSC format for sampling:
        to_rel_type = {k: '__'.join(k) for k in data.edge_types}
        to_edge_type = {v: k for k, v in to_rel_type.items()}
        colptr_dict, row_dict, perm = to_hetero_csc(
            data, device=self.device, share_memory=self.share_memory,
            is_sorted=self.is_sorted)

        row_dict = remap_keys(row_dict, to_rel_type)
        colptr_dict = remap_keys(colptr_dict, to_rel_type)


        node, row, col, edge, batch, num_sampled_nodes, num_sampled_edges = self.sample_hetero_graph(
            data, target_nodes, colptr_dict, row_dict, to_edge_type)

        # filter data to create a new HeteroData object.
        data_out = filter_hetero_data(data, node, row, col, edge, None)


        if WITH_PYG_LIB and self.fetch_neighbors:
            self.set_neighbor_mask_node(data_out, num_sampled_nodes)
            self.set_neighbor_mask_edge(data_out, num_sampled_edges)

        # for k, v in target_lenghts.items():
        #     data_out[k].num_sampled_nodes = v

        return data_out, len(target_nodes["note"])

    def sample_hetero_graph(self, data, target_nodes, colptr_dict, row_dict, to_edge_type):
        # Sample subgraph:
        if WITH_PYG_LIB:
            args = (
                data.node_types,
                data.edge_types,
                colptr_dict,
                row_dict,
                target_nodes,  # seed_dict
                self.num_neighbors.get_mapped_values(data.edge_types),
                self.node_time,
            )
            # Setting to None
            seed_time = None
            args += (seed_time,)
            if torch_geometric.typing.WITH_WEIGHTED_NEIGHBOR_SAMPLE:
                args += (self.edge_weight,)
            args += (
                True,  # csc
                False,  # do not replace
                True,  # Subgraph not induced
                False,  # not disjoint
                'uniform',  # temporal strategy
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
                True,  # subgraph not induced
                False,  # do not replace
            )
            node, row, col, edge, batch = out + (None,)

        # `pyg-lib>0.1.0` returns sampled number of nodes/edges:
        num_sampled_nodes = num_sampled_edges = None
        if len(out) >= 6:
            num_sampled_nodes, num_sampled_edges = out[4:6]
        row = remap_keys(row, to_edge_type)
        col = remap_keys(col, to_edge_type)
        edge = remap_keys(edge, to_edge_type)
        return node, row, col, edge, batch, num_sampled_nodes, num_sampled_edges


    @property
    def num_neighbors(self) -> NumNeighbors:
        return self._num_neighbors

    @num_neighbors.setter
    def num_neighbors(self, num_neighbors: NumNeighborsType):
        if isinstance(num_neighbors, NumNeighbors):
            self._num_neighbors = num_neighbors
        else:
            self._num_neighbors = NumNeighbors(num_neighbors)

    def set_neighbor_mask_node(self, data, num_sampled_nodes):
        for key, value in num_sampled_nodes.items():
            neighbor_mask = torch.zeros(data[key].x.shape[0], dtype=torch.long, device=data[key].x.device)
            value = np.cumsum(value)
            assert value[-1] == data[key].x.shape[0]
            for i in range(1, len(value)):
                neighbor_mask[value[i - 1]:value[i]] = i
            data[key].neighbor_mask = neighbor_mask

    def set_neighbor_mask_edge(self, data, num_sampled_edges):
        for key, value in num_sampled_edges.items():
            key = tuple(key.split("__")) if isinstance(key, str) else key
            neighbor_mask = torch.zeros(data[key].edge_index.shape[1], dtype=torch.long, device=data[key].edge_index.device)
            value = np.cumsum(value)
            assert value[-1] == data[key].edge_index.shape[1]
            for i in range(1, len(value)):
                neighbor_mask[value[i - 1]:value[i]] = i
            data[key].neighbor_mask = neighbor_mask

    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'