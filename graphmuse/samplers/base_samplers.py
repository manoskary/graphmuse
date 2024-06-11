import numpy as np
from torch.utils.data import Sampler
import torch


class SubgraphMultiplicitySampler(Sampler):
    """
    This sampler is used to create subgraphs from a graph dataset.

    This sampler takes as input list of graphs and creates returns a multiplicity of their indices base on their size.
    It creates a different number of subgraphs based on the size of the original graph and the max subgraph size.
    It is used in the ASAPGraphDataset class.
    """
    def __init__(self, data_source, max_subgraph_size=100, drop_last=False, batch_size=64, multiplicity_ratio:float=2):
        self.data_source = data_source
        bucket_boundaries = [2*max_subgraph_size, 5*max_subgraph_size, 10*max_subgraph_size, 20*max_subgraph_size]
        self.sampling_sizes = (np.array([2, 4, 10, 20, 40])*multiplicity_ratio).astype(int)
        self.ind_n_len = [(i, num_nodes) for i, num_nodes in enumerate(data_source)]
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