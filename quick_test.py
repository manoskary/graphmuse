import graphmuse.samplers as sam

from tqdm import tqdm
import os
import partitura as pt
import numpy as np    

def test_preneighborhood_count():
    N = 10

    edges = np.empty((2, N*(N+1)//2), dtype=np.uint32)

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
# from tests import test_samplers, test_graph_creation

# ts = test_samplers.TestSamplers()

# ts.test_nodewise_sampling(sam.sample_nodewise)
# ts.test_nodewise_sampling(sam.sample_nodewise_mt_static)

# test_graph_creation.TestGraphMuse().test_edge_list()

# onsets = numpy.random.randint(0,10,20)
# onsets.sort()
# na = {"onset_div": onsets}
# print(onsets)
# print(sam.random_score_region(na, 5))

score_path = os.path.join(os.path.dirname(__file__), "tests", "samples", "wtc1f01.musicxml")
score = pt.load_score(score_path)
note_array = score.note_array()

edge_list, _ = sam.compute_edge_list(note_array['onset_div'].astype(np.int32), note_array['duration_div'].astype(np.int32))

perm=edge_list[1,:].argsort()
edge_list = edge_list[:,perm]
_, uniq_indices = np.unique(edge_list[1,:], return_index=True)

for i in range(len(uniq_indices)-1):
    edge_list[0,uniq_indices[i]:uniq_indices[i+1]].sort()

edge_list[0, uniq_indices[-1]:].sort()

#print((edge_list[1][:-1]<=edge_list[1][1:]).all())

g = sam.graph(edge_list)

region = sam.random_score_region(note_array, 100)

print(region)

print(sam.extend_score_region_via_neighbor_sampling(g, note_array, region, 2))