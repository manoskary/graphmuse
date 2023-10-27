import numpy as np
from torch.utils.data import DataLoader, Sampler
import torch
import graphmuse.samplers as csamplers


class SubgraphCreationSampler(Sampler):
    """
    This sampler takes as input a graph dataset and creates subgraphs of a given size.
    It creates a different number of subgraphs based on the size of the original graph and the max subgraph size.
    It is used in the ASAPGraphDataset class.
    """
    def __init__(self, data_source, max_subgraph_size=100, drop_last=False, batch_size=64, train_idx=None, subgraphs_per_max_size:int=5):
        self.data_source = data_source
        bucket_boundaries = [2*max_subgraph_size, 5*max_subgraph_size, 10*max_subgraph_size, 20*max_subgraph_size]
        self.sampling_sizes = np.array([2, 4, 10, 20, 40])*subgraphs_per_max_size
        ind_n_len = []
        train_idx = train_idx if train_idx is not None else list()
        for i, g in enumerate(data_source.graphs):
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
	def __init__(self, graphs, subgraph_size, subgraphs, num_layers=0, samples_per_node=0, batch_size=1, num_workers=0):
		self.graphs = graphs
		gp = numpy.array([g.node_count() for g in graphs])
		self.graph_probabilities = gp/numpy.sum(gp)
		self.subgraph_size = subgraph_size
		self.subgraphs = subgraphs
		self.num_layers = num_layers # This is for a later version with node-wise sampling
		self.samples_per_node = samples_per_node
		self.onsets = {}
		self.onset_count = {}
		batch_sampler = SubgraphCreationSampler(self, max_subgraph_size=subgraph_size, drop_last=False, batch_size=batch_size)
		super().__init__(batch_sampler=batch_sampler, batch_size=1, collate_fn=self.collate_fn, num_workers=num_workers)

	def collate_fn(self, batch):
		graphlist = self.graphs[batch]
		out = self.sample_from_graphlist(graphlist, self.subgraph_size, self.subgraphs, self.num_layers)
		return out

	def sample_from_graphlist(self, graphlist, subgraph_size, subgraphs, num_layers=None):
		"""
		Sample subgraphs from a list of graphs.
		"""

		subgraph_samples = []

		for _ in range(self.subgraphs):
			random_graph = np.choice(self.graphs, p=self.graph_probabilities)

			region = csamplers.random_score_region(random_graph.note_array, self.subgraph_size)

			(left_extension, left_edges), (right_ext, right_edges) = csamplers.extend_score_region_via_neighbor_sampling(random_graph.c_graph, random_graph.note_array, region, self.samples_per_node)

			left_layers, edge_indices_between_left_layers, _ = csamplers.sample_nodewise(random_graph.c_graph, self.num_layers-2, self.samples_per_node, left_extension)
			
			edges_between_left_layers = [random_graph.edges[idx] for idx in edge_indices_between_left_layers]


			right_layers, edges_between_right_layers = csamplers.sample_neighbors_in_score_graph(random_graph.note_array, self.num_layers-2, self.samples_per_node, right_extension)
			
			layers=[numpy.arange(region[0], region[1])]

			layers.extend([np.concatenate([l,r]) for l,r in zip(left_layers, right_layers)])

			edges_between_layers = [np.concatenate([left_edges, right_edges])]
			edges_between_layers.extend([np.concatenate([l,r]) for l,r in zip(edges_between_left_layers, edge_indices_between_right_layers)])

			subgraph_samples.append((layers, edges_between_layers))


		return subgraph_samples

