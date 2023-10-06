import numpy as np
from torch.utils.data import DataLoader, Sampler
import torch


class SubgraphCreationSampler(Sampler):
    """
    This sampler is used to create subgraphs from a graph dataset.

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
	def __init__(self, graphs, subgraph_size, subgraphs, num_layers=None, batch_size=1, num_workers=0):
		self.graphs = graphs
		self.subgraph_size = subgraph_size
		self.subgraphs = subgraphs
		self.num_layers = num_layers # This is for a later version with node-wise sampling
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
		return

	def random_score_region(self, graph_idx, check_possibility=True):
		if graph_idx in self.onsets.keys():
			onsets = self.onsets[graph_idx]
			onset_count = self.onset_count[graph_idx]
		else:
			onsets = self.graphs.note_array['onset_div'].astype(np.int32)
			uniques, onset_count = np.unique(onsets, return_counts=True)
			self.onsets[graph_idx] = onsets
			self.onset_count[graph_idx] = onset_count

		# in order to avoid handling the special case where a region is sampled that reaches to the end of 'onsets', we simply extend the possible values
		indices = np.concatenate([self.subgraph_size,[len(onsets)]])

		if check_possibility:
			if (np.diff(indices)>self.subgraph_size).all():
				raise ValueError("by including all notes with the same onset, the budget is always exceeded")

		# since we added the last element ourselves and it isn't a valid index,
		# we only sample excluding the last element
		# using a random permutation isn't necessarily, it just avoids sampling a previous sample
		for idx in np.random.permutation(len(indices)-1):
			samples_start = indices[idx]

			if samples_start+self.subgraph_size>=len(onsets):
				return (samples_start,len(onsets))

			samples_end = samples_start+self.subgraph_size

			while samples_end-1>=samples_start and onsets[samples_end]==onsets[samples_end-1]:
				samples_end-=1

			if samples_start<samples_end:
				return (samples_start, samples_end)


		if check_possibility:
			assert False, "a result should be possible, according to the check above, however, no result exists."
		else:
			raise ValueError("by including all notes with the same onset, the budget is always exceeded")

	def musical_sampling(self, g_idxs, check_possibility=True):
		# we want to sample from the array 'graphs' proportional to the size of the graphs in the array
		# so we need to pre-compute a probability distribution for that
		graphs = [self.graphs[i] for i in g_idxs]
		subgraphs = []
		for i,g in enumerate(graphs):

			if g.size() <= self.subgraph_size:
				(l, r) = (0, g.size())
			else:
				(l, r) = self.random_score_region(g_idxs[i], check_possibility)
				assert r - l <= self.subgraph_size

			subgraphs.append((g_idxs[i], (l, r)))

		return subgraphs

