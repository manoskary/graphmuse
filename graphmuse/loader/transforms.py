import torch
from torch_geometric.data import Batch


def transform_to_pyg(data: Batch, num_hops: int) -> Batch:
    r"""Transforms the data in the batch to the PYG Hierarchical sampling format."""
    remap = {k: torch.cat([torch.where(data[k].neighbor_mask == i)[0] for i in range(num_hops + 1)],
                          dim=0) for k in data.node_types}
    for k in data.node_types:
        data[k].x = data[k].x[remap[k]]
        shape_ckecker = data[k].x.shape[0]
        data[k].num_sampled_nodes = torch.bincount(data[k].neighbor_mask)
        data[k].batch_size = data[k].num_sampled_nodes[0]
        data[k].pop("neighbor_mask")
        for attribute in data[k].keys():
            # if the attribute size is the same as the number of nodes, we can remap it
            if attribute not in ["x", "num_sampled_nodes", "batch_size"]:
                if isinstance(data[k][attribute], torch.Tensor):
                    if data[k][attribute].shape[0] == shape_ckecker:
                        data[k][attribute] = data[k][attribute][remap[k]]

    for k in data.edge_types:
        src_type = k[0]
        dst_type = k[-1]
        # first remap the edge index and then reorder it
        edge_index = data[k].edge_index
        edge_index[0] = remap[src_type][edge_index[0]]
        edge_index[1] = remap[dst_type][edge_index[1]]
        data[k].edge_index = edge_index
        # Now we can reorder the edge index and the edge attributes
        data[k].edge_index = edge_index[:, torch.cat(
            [torch.where(data[k].neighbor_mask == i)[0] for i in range(num_hops)], dim=0)]
        shape_ckecker = data[k].edge_index.shape[1]
        data[k].num_sampled_edges = torch.bincount(data[k].neighbor_mask)
        data[k].pop("neighbor_mask")
        for attribute in data[k].keys():
            # if the attribute size is the same as the number of nodes, we can remap it
            if attribute not in ["edge_index", "num_sampled_edges"]:
                if isinstance(data[k][attribute], torch.Tensor):
                    if data[k][attribute].shape[1] == shape_ckecker:
                        data[k][attribute] = data[k][attribute][:, remap[k]]
    return data
