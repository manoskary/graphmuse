import graphmuse.samplers as sam
import unittest
import numpy as np


class TestSamplers(unittest.TestCase):

    def test_nodewise_sampling(self):
        for nodewise_sampling_method in (sam.sample_nodewise,):
            print(f"Unit Testing for {nodewise_sampling_method.__name__}")

            V = 100
            E = np.random.randint(1,V**2,1)[0]

            edges = np.random.randint(0,V,(2, E),dtype=np.int32)

            V = np.max(edges)

            resort_idx = np.lexsort((edges[0], edges[1]))
            edges = edges[:, resort_idx]

            g = sam.graph(edges)

            target_size = np.random.randint(1,V//4,1)[0]
            target = np.unique(np.random.randint(0, V, target_size, np.uint32))
            depth = 3
            samples_per_node = 3
            samples_per_layer, edges_between_layers, load_per_layer = nodewise_sampling_method(g, depth, samples_per_node, target)

            self.assertTrue(len(samples_per_layer)==depth+1)
            self.assertTrue(len(edges_between_layers)==depth)
            self.assertTrue(len(load_per_layer)==depth)
            self.assertTrue(samples_per_layer[-1].shape == target.shape)
            self.assertTrue((sorted(samples_per_layer[-1])==sorted(target)))

            for l in range(depth):
                self.assertTrue(set(samples_per_layer[l]).union(samples_per_layer[l+1]) == set(load_per_layer[l]))
                current_edges = edges_between_layers[l]
                current_edges_list = [(current_edges[0,i],current_edges[1,i]) for i in range(current_edges.shape[1])]


                samples_counter = dict()

                for s,d in current_edges_list:
                    if d in samples_counter.keys():
                        samples_counter[d]+=1
                    else:
                        samples_counter[d]=1

                for d,c in samples_counter.items():
                    self.assertTrue(c == min(samples_per_node,g.preneighborhood_count(d)), f"count of {d} is {c}, but pnc is {g.preneighborhood_count(d)}")

                unique_src = np.unique(current_edges[0])

                self.assertTrue(unique_src.shape == samples_per_layer[l].shape)
                self.assertTrue(list(unique_src) == sorted(samples_per_layer[l]))

                unique_dst = set(current_edges[1])

                for sample in samples_per_layer[l+1]:
                    if sample not in unique_dst:
                        self.assertTrue(g.preneighborhood_count(sample)==0)

            print(f"{nodewise_sampling_method.__name__} passed all tests")
