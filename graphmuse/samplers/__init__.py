from .csamplers import *
import numpy

def graph(edges):
	if edges.dtype != numpy.uint32:
		raise TypeError("currently only numpy.uint32 nodes supported")

	node_count = max(numpy.max(edges[0]), edges[1][-1])+1

	# TODO: change strides such that src and dst can be iterated separately without strides in C code

	return Graph(edges, node_count)

def random_score_region(note_array, budget):
	onsets = note_array["onset_div"].astype(numpy.int32)
	_,unique_onset_indices = numpy.unique(onsets, return_index=True)

	unique_onset_indices = unique_onset_indices.astype(numpy.uint32)

	if len(onsets) - unique_onset_indices[-1] > budget and (numpy.diff(unique_onset_indices)>budget).all():
		raise ValueError("impossible to sample a score region with the given budget within given note array")

	return c_random_score_region(onsets, unique_onset_indices, budget)

def extend_score_region_via_neighbor_sampling(graph, note_array, region, samples_per_node):
	region_start, region_end = region
	onsets = note_array["onset_div"].astype(numpy.int32)
	durations = note_array["duration_div"].astype(numpy.int32)
	

	if region_start>=region_end:
		raise ValueError("invalid region given")

	if region_start==0 and region_end==len(onsets)-1:
		raise ValueError("can't extend score region if the region covers the entire score")


	endtimes_cummax = numpy.maximum.accumulate(onsets+durations)

	return c_extend_score_region_via_neighbor_sampling(graph,onsets, durations, endtimes_cummax, region_start, region_end, samples_per_node)

def sample_neighbors_in_score_graph(note_array, depth, samples_per_node, targets):
	assert len(targets)>0

	onsets = note_array["onset_div"].astype(numpy.int32)
	durations = note_array["duration_div"].astype(numpy.int32)

	return c_sample_neighbors_in_score_graph(onsets, durations, depth, samples_per_node, targets)