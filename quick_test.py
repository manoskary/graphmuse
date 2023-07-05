import graphmuse.samplers as sam
import numpy
from tqdm import tqdm
import numpy
    

def test_preneighborhood_count():
    N = 10

    edges = numpy.empty((2, N*(N+1)//2), dtype=numpy.uint32)

    cursor = 0

    for i in range(N):
        for j in range(N-i-1,-1,-1):
            edges[0, cursor]=j
            edges[1, cursor]=i
            cursor+=1

    graph = sam.graph(edges)

    graph.print()

    for i in range(N):
        assert graph.preneighborhood_count(i)==N-i

#test_preneighborhood_count()
from tests import test_samplers, test_graph_creation

ts = test_samplers.TestSamplers()

ts.test_nodewise_sampling(sam.sample_nodewise)
ts.test_nodewise_sampling(sam.sample_nodewise_mt_static)

test_graph_creation.TestGraphMuse().test_edge_list()