import graphmuse.samplers as sam
import numpy
from tqdm import tqdm

def test_nodewise_sampling(nodewise_sampling_method):
    print(f"unit testing {nodewise_sampling_method.__name__}")

    V = 100
    E = numpy.random.randint(1,V**2,1)[0]

    edges = numpy.random.randint(0,V,(2, E),dtype=numpy.uint32)

    edges = sorted(list(set([(edges[0,i], edges[1,i]) for i in range(E)])), key=lambda t:t[1])

    edges = numpy.array(edges).T

    g = sam.graph(edges)

    target_size = numpy.random.randint(1,V//4,1)[0]

    target = numpy.random.randint(0, V, target_size, numpy.uint32)

    print("And the targets are: ", target)

    depth = 3
    samples_per_node = 3

    samples_per_layer, edge_indices_between_layers, load_per_layer = nodewise_sampling_method(g, depth, samples_per_node, target)



    assert(len(samples_per_layer)==depth+1)
    assert(len(edge_indices_between_layers)==depth)
    assert(len(load_per_layer)==depth)

    assert(samples_per_layer[-1].shape == target.shape)
    assert((sorted(samples_per_layer[-1])==sorted(target)))

    for l in tqdm(range(depth)):

        assert(set(samples_per_layer[l]).union(samples_per_layer[l+1]) == set(load_per_layer[l]))

        current_edges = edges[:,edge_indices_between_layers[l]]

        current_edges_list = [(current_edges[0,i],current_edges[1,i]) for i in range(current_edges.shape[1])]

        samples_counter = dict()

        for s,d in current_edges_list:
            if d in samples_counter.keys():
                samples_counter[d]+=1
            else:
                samples_counter[d]=1

        for d,c in samples_counter.items():
            assert(c==min(samples_per_node,g.preneighborhood_count(d)))

        # print(samples_per_layer[l])
        # print(edges[:,numpy.sort(edge_indices_between_layers[l])])
        # print(samples_per_layer[l+1])
        # print("_________________________________________________________________")

        unique_src = numpy.unique(current_edges[0])

        assert(unique_src.shape == samples_per_layer[l].shape)
        assert(list(unique_src) == sorted(samples_per_layer[l]))

        unique_dst = numpy.unique(current_edges[1])

        assert(unique_dst.shape == samples_per_layer[l+1].shape), f"{(set(unique_dst)-set(samples_per_layer[l+1])).union(set(samples_per_layer[l+1])-set(unique_dst))}"
        assert(list(unique_dst) == sorted(samples_per_layer[l+1]))

        





    print(f"{nodewise_sampling_method.__name__} passed unit tests")



test_nodewise_sampling(sam.sample_nodewise)

# from tests import test_samplers

# test_samplers.TestSamplers().test_nodewise_sampling(sam.sample_nodewise)
#test_samplers.TestSamplers().test_nodewise_sampling(sam.sample_nodewise)

# from tests import test_graph_creation
# print("test edge list:")
# test_graph_creation.TestGraphMuse().test_edge_list()


x=numpy.empty(10)
print(x)