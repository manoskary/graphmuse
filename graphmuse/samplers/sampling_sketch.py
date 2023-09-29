import numpy as np


class Sampler:
	def __init__(self, graphs, subgraph_size, subgraphs, num_layers=None):
		self.graphs = graphs
		self.subgraph_size = subgraph_size
		self.subgraphs = subgraphs
		self.num_layers = num_layers # This is for a later version with node-wise sampling
		self.onsets = {}
		self.onset_count = {}

	def prepare_data(self):
		graph_sizes = np.array([g.num_nodes for g in self.graphs])
		multiples = graph_sizes // self.subgraph_size + 1
		indices = np.concatenate([np.repeat(i, m) for i, m in enumerate(multiples)])
		# returns a list of indices with repeats have to look to documentation for precise format.

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

