import graphmuse.samplers as sam
import unittest
from tqdm import tqdm
import numpy


class TestSamplers(unittest.TestCase):

    def test_nodewise_sampling(self, nodewise_sampling_method):
        print(f"unit testing {nodewise_sampling_method.__name__}")

        V = 100
        E = numpy.random.randint(1,V**2,1)[0]

        edges = numpy.random.randint(0,V,(2, E),dtype=numpy.uint32)

        edges = sorted(list(set([(edges[0,i], edges[1,i]) for i in range(E)])), key=lambda t:t[1])

        V = numpy.max(edges)

        #print(V)

        edges = numpy.array(edges).T

        g = sam.graph(edges)

        # print(edges)
        # g.print()

        target_size = numpy.random.randint(1,V//4,1)[0]

        target = numpy.unique(numpy.random.randint(0, V, target_size, numpy.uint32))

        #print("And the targets are: ", target)

        depth = 3
        samples_per_node = 3

        samples_per_layer, edge_indices_between_layers, load_per_layer = nodewise_sampling_method(g, depth, samples_per_node, target)

        assert(len(samples_per_layer)==depth+1)
        assert(len(edge_indices_between_layers)==depth)
        assert(len(load_per_layer)==depth)

        assert(samples_per_layer[-1].shape == target.shape)
        assert((sorted(samples_per_layer[-1])==sorted(target)))

        for l in range(depth):
            # print("layer ",l)
            # print(samples_per_layer[l])
            # print(edges[:,numpy.sort(edge_indices_between_layers[l])])
            # print(samples_per_layer[l+1])
            # print("_________________________________________________________________")

            assert(set(samples_per_layer[l]).union(samples_per_layer[l+1]) == set(load_per_layer[l]))

            current_edges = edges[:,edge_indices_between_layers[l]]

            current_edges_list = [(current_edges[0,i],current_edges[1,i]) for i in range(current_edges.shape[1])]

            #print(sorted(current_edges_list, key=lambda t:t[1]))

            samples_counter = dict()

            for s,d in current_edges_list:
                if d in samples_counter.keys():
                    samples_counter[d]+=1
                else:
                    samples_counter[d]=1

            for d,c in samples_counter.items():
                assert(c==min(samples_per_node,g.preneighborhood_count(d))),f"count of {d} is {c}, but pnc is {g.preneighborhood_count(d)}"

            

            unique_src = numpy.unique(current_edges[0])

            assert(unique_src.shape == samples_per_layer[l].shape)
            assert(list(unique_src) == sorted(samples_per_layer[l]))

            unique_dst = set(current_edges[1])

            for sample in samples_per_layer[l+1]:
                if sample not in unique_dst:
                    assert g.preneighborhood_count(sample)==0

            





        print(f"{nodewise_sampling_method.__name__} passed unit tests")