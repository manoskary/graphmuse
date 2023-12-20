import graphmuse.samplers as sam
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

g = sam.graph(edge_list)

region = sam.random_score_region(note_array, 100)

print(region)

sam.c_set_seed(10101)

region = sam.random_score_region(note_array, 100)

print(region)

(left_ext, left_edges), (right_ext, right_edges) = sam.extend_score_region_via_neighbor_sampling(g, note_array, region, 2)


print(left_ext)
print()
print(left_edges)
print("\n----------------------\n")
print(right_ext)
print()
print(right_edges)
print("\n----------------------\n")





right_right_ext, right_right_edges,_ = sam.sample_neighbors_in_score_graph(note_array, 1, 3, right_ext.numpy())

print(right_right_ext)
print()
print(right_right_edges)