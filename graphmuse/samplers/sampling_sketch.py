import numpy as np
from torch.utils.data import DataLoader, Sampler
import torch
import graphmuse.samplers as csamplers
from torch_geometric.data import Data, Batch


class SubgraphCreationSampler(Sampler):
    """
    This sampler takes as input a graph dataset and creates subgraphs of a given size.
    It creates a different number of subgraphs based on the size of the original graph and the max subgraph size.
    It is used in the ASAPGraphDataset class.

    Parameters
    ----------
    data_source : list of HeteroScoreGraph
        The graph dataset.
    max_subgraph_size : int
        The maximum size of the subgraphs.
    drop_last : bool
        Whether to drop the last batch if it is smaller than the batch size.
    batch_size : int
        The batch size.
    train_idx : list of int
        The indices of the training graphs.
    subgraphs_per_max_size : int
        The number of subgraphs to create for each max size.
    """
    def __init__(self, graphs, max_subgraph_size=100, drop_last=False, batch_size=64, train_idx=None, subgraphs_per_max_size:int=5):
        self.data_source = graphs
        bucket_boundaries = [2*max_subgraph_size, 5*max_subgraph_size, 10*max_subgraph_size, 20*max_subgraph_size]
        self.sampling_sizes = np.array([2, 4, 10, 20, 40])*subgraphs_per_max_size
        ind_n_len = []
        train_idx = train_idx if train_idx is not None else range(len(graphs))
        for i, g in enumerate(graphs):
            if i in train_idx:
                ind_n_len.append((i, g.x.shape[0]))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.drop_last = drop_last
        self.batch_size = batch_size
        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")
        self.max_size = max_subgraph_size
        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)
        self.reindex_data()

    def reindex_data(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid] += [p]*self.sampling_sizes[pid].item()
            else:
                data_buckets[pid] = [p]*self.sampling_sizes[pid].item()
        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])
        self.indices = torch.cat(list(data_buckets.values()))
        self.num_samples = len(self.indices)

    def __iter__(self):
        idx = self.indices[torch.multinomial(torch.ones(self.num_samples), self.num_samples, replacement=False)]
        batch = torch.split(idx, self.batch_size, dim=0)
        if self.drop_last and len(batch[-1]) != self.batch_size:
            batch = batch[:-1]
        for i in batch:
            yield i.numpy().tolist()

    def __len__(self):
        return self.num_samples // self.batch_size

    def element_to_bucket_id(self, x, seq_length):
        valid_buckets = (seq_length >= self.buckets_min) * (seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()
        return bucket_id


class MuseDataloader(DataLoader):
    """
    This dataloader takes as input a list of graphs and creates subgraphs of a given size.
    It creates a different number of subgraphs based on the size of the original graph and the max subgraph size.

    Parameters
    ----------
    graphs : list of HeteroScoreGraph
        The graph dataset.
    subgraph_size : int
        The maximum size of the subgraphs.
    subgraphs : int
        The number of subgraphs per batch. Maybe it is redundant and it is covered by the batch size.
    num_layers : int
        The number of layers to sample.
    samples_per_node : int
        The number of samples per node.
    batch_size : int
        The batch size.
    num_workers : int
        The number of workers.
    """
    def __init__(self, graphs, subgraph_size, num_layers=3, samples_per_node=3, batch_size=1, num_workers=0, sample_rightmost=False, device="cpu", verbose=False, **kwargs):
        self.graphs = graphs
        self.subgraph_size = subgraph_size
        self.device = device
        self.random_state = kwargs.get("random_state", None)
        if self.random_state is not None:
            csamplers.c_set_seed(self.random_state)
            if verbose:
                print("Setting random state to {}".format(self.random_state))
        self.sample_rightmost = sample_rightmost
        self.num_layers = num_layers # This is for a later version with node-wise sampling
        self.samples_per_node = samples_per_node
        self.etypes = {
            "onset": 0,
            "consecutive": 1,
            "during": 2,
            "silence": 3,
        }
        self.onsets = {}
        self.onset_count = {}
        dataset = range(len(graphs))
        batch_sampler = SubgraphCreationSampler(graphs, max_subgraph_size=subgraph_size, drop_last=False, batch_size=batch_size)
        super().__init__(dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=self.collate_graph_fn)

    def collate_graph_fn(self, batch):
        graphlist = [self.graphs[idx] for idx in batch]
        out = self.sample_from_graphlist(graphlist)
        return out

    def sample_from_graphlist(self, graphlist):
        """
        Sample subgraphs from a list of graphs.
        This method samples a subgraph from each graph in the list.
        """
        subgraph_samples = []

        # Given a list of graphs, sample a subgraph from each graph of size at most subgraph_size
        for random_graph in graphlist:
            region = csamplers.random_score_region(random_graph.note_array, self.subgraph_size)

            # NOTE: This needs correction
            sampled_nodes_first_layer, edges_within_region = csamplers.sample_preneighbors_within_region(random_graph.c_graph, region, self.samples_per_node)#torch.tensor([]).long(), torch.tensor([[],[]]).long()
            # sampled_nodes_first_layer, edges_within_region = torch.tensor([]).long(), torch.tensor([[], []]).long()

            # This is the last layer neighbors extension
            # (left_extension, left_edges), (right_extension, right_edges) = csamplers.extend_score_region_via_neighbor_sampling(random_graph.c_graph, random_graph.note_array, region, self.samples_per_node, self.sample_rightmost)
            (left_extension, left_edges), (right_extension, right_edges) = (torch.tensor([]).long(), torch.tensor([[], []]).long()), (torch.tensor([]).long(), torch.tensor([[], []]).long())

            # Sample leftmost typical node-wise by num layers excluding the last layer which was sampled above
            left_layers, edges_between_left_layers, _, total_left_samples = csamplers.sample_nodewise(random_graph.c_graph, self.num_layers-2, self.samples_per_node, left_extension.numpy())
            # left_layers, edges_between_left_layers, total_left_samples = torch.tensor([]).long(), torch.tensor([[], []]).long(), torch.tensor([]).long()

            if self.sample_rightmost:
                # Sample rightmost node-wise by num layers (because of reverse edges missing)
                right_layers, edges_between_right_layers, total_right_samples = csamplers.sample_neighbors_in_score_graph(random_graph.note_array, self.num_layers-2, self.samples_per_node, right_extension.numpy())
            else:
                right_layers, edges_between_right_layers, total_right_samples = torch.tensor([]).long(), torch.tensor([[], []]).long(), torch.tensor([]).long()

            cat_list = (edges_within_region, left_edges, right_edges)

            # filtering out edge types
            # NOTE: We need to filter out the edge types here (not done for the right layers)
            cat_list = [e[:2] for e in cat_list]

            # Translate edges to subgraph indices (do this on GPU when available).
            subgraph_edge_index = torch.cat(cat_list, dim=1)
            sampled_nodes = torch.cat((torch.arange(region[0], region[1]), sampled_nodes_first_layer, total_left_samples, total_right_samples)).to(self.device)
            new_mapping = torch.arange(sampled_nodes.shape[0], device=self.device)
            nodes_remap = torch.empty_like(torch.zeros(random_graph.x.shape[0]), dtype=int).to(self.device)
            nodes_remap[sampled_nodes] = new_mapping
            # Map the indices of the subgraph to the indices of the original graph
            new_edge_index = nodes_remap[subgraph_edge_index]
            # Create a PyG graph
            subgraph_samples.append(
                Data(x=random_graph.x[sampled_nodes].to(self.device), edge_index=new_edge_index).to(self.device))

        # Join all subgraphs together
        batch = Batch.from_data_list(subgraph_samples)

        return batch

    def __iter__(self):
        return super().__iter__()
