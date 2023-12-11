from .csamplers import *
import numpy
import torch

def graph(edges):
	"""
	creates a Graph object from a numpy array of edges.
	It includes type checking.

	Parameters
	----------
	edges : numpy.ndarray
		a 2D array of integers. The first row contains the source nodes, the second row the destination nodes, the third row the edge types.

	Returns
	-------
	Graph
		a Graph object in C

	"""
	if edges.dtype not in (numpy.int64, numpy.int32):
		raise TypeError(f"currently only {numpy.int32} and {numpy.int64} nodes supported, not {edges.dtype}")

	node_count = max(numpy.max(edges[0]), edges[1][-1])+1

	# TODO: change strides such that src and dst can be iterated separately without strides in C code

	return Graph(edges, node_count)


def sample_nodewise(cgraph, depth, samples_per_node, targets):
	"""
	Python wrapper function for C Extension function c_sample_nodewise
	Samples nodes within a score graph

	Parameters
	----------
	cgraph : Graph
		The score graph implemented in c. It is an attribute of the HeteroScoreGraph.
	depth : int
		The number of layers that are sampled
	samples_per_node : int
		The number of samples per node.
	targets : np.ndarray
		initial value for the sampling iteration
	"""
	samples_per_layer, edges_between_layers, load_per_layer, total_samples = c_sample_nodewise(cgraph, depth, samples_per_node, targets)
	# move to torch tensors
	samples_per_layer = [torch.from_numpy(layer) for layer in samples_per_layer]
	edges_between_layers = [torch.from_numpy(edges) for edges in edges_between_layers]
	load_per_layer = [torch.from_numpy(layer) for layer in load_per_layer]
	total_samples = torch.from_numpy(total_samples)

	return samples_per_layer, edges_between_layers, load_per_layer, total_samples


def random_score_region(note_array, budget):
	"""
	Python wrapper function for C Extension function c_random_score_region

	It samples a score region of a given budget from a score graph.

	Parameters
	----------
	note_array : partitura/numpy.structured array
		This represents a score graph, as in, the data in this structure determines the edges between nodes
		required fields: onset_div, duration_div
		note_array['onset_div'] is a non-decreasing integer array
		note_array['duration_div'] is an integer array
	budget : int
		The maximum number of nodes in the region

	Returns
	-------
	region : tuple
		The region to sample from. It is a tuple of two integers, start and end.
	"""
	onsets = note_array["onset_div"].astype(numpy.int32)
	_,unique_onset_indices = numpy.unique(onsets, return_index=True)

	unique_onset_indices = unique_onset_indices.astype(numpy.int32)

	if len(onsets) - unique_onset_indices[-1] > budget and (numpy.diff(unique_onset_indices)>budget).all():
		raise ValueError("impossible to sample a score region with the given budget within given note array")

	return c_random_score_region(unique_onset_indices, budget)


def extend_score_region_via_neighbor_sampling(cgraph, note_array, region, samples_per_node, sample_rightmost=True):
	"""
	Python wrapper function for C Extension function c_extend_score_region_via_neighbor_sampling
	Samples neighbors and pre-neighbors of nodes withinin a score region such that it only considers the neighbors and pre-neighbors oustide the region

	Parameters
	----------
	cgraph : Graph
		The score graph implemented in c. It is an attribute of the HeteroScoreGraph.
	note_array : partitura/numpy.structured array
		This represents a score graph, as in, the data in this structure determines the edges between nodes
		required fields: onset_div, duration_div
		note_array['onset_div'] is a non-decreasing integer array
		note_array['duration_div'] is an integer array
	region : tuple
		The region to sample from. It is a tuple of two integers, start and end.
		The region is inclusive on the left and exclusive on the right.
	samples_per_node : int
		The number of samples per node.
	sample_rightmost : bool
		flag that determines whether or not to compute the right extension

	Note: 	c_extend_score_region_via_neighbor_sampling expects 'onsets' and 'durations' to be passed in as int32 integer arrays
			furthermore, c_extend_score_region_via_neighbor_sampling expects the cumulative maximum of 'onsets+durations'
			furthermore, c_extend_score_region_via_neighbor_sampling expects the region to be passed as 2 separate integers
			(see code below)

	Returns
	-------
	left_extension : Tuple()
		(left_nodes, left_edges)
		left_nodes: np.ndarray
			1D array of integers. The sampled nodes.
		left_edges: np.ndarray (2, num_edges)
			2D array of integers. The edges. The first row contains the source nodes, the second row the destination nodes.
	right_extension : Tuple()
		(right_nodes, right_edges)
		right_nodes: np.ndarray
			1D array of integers. The sampled nodes.
		right_edges: np.ndarray (2, num_edges)
			2D array of integers. The edges. The first row contains the source nodes, the second row the destination nodes.

	"""

	region_start, region_end = region
	onsets = note_array["onset_div"].astype(numpy.int32)
	durations = note_array["duration_div"].astype(numpy.int32)
	
	if not isinstance(sample_rightmost, bool):
		raise ValueError(f"non-bool object {sample_rightmost} passed in as sample_rightmost parameter")

	if region_start>=region_end:
		raise ValueError("invalid region given")

	if region_start==0 and region_end==len(onsets)-1:
		raise ValueError("can't extend score region if the region covers the entire score")


	endtimes_cummax = numpy.maximum.accumulate(onsets+durations)
	(left_nodes, left_edges), (right_nodes, right_edges) = c_extend_score_region_via_neighbor_sampling(cgraph, onsets, durations, endtimes_cummax, region_start, region_end,
												samples_per_node, sample_rightmost)

	# move to torch tensors
	left_nodes = torch.from_numpy(left_nodes)
	left_edges = torch.from_numpy(left_edges)
	right_nodes = torch.from_numpy(right_nodes)
	right_edges = torch.from_numpy(right_edges)

	return (left_nodes, left_edges), (right_nodes, right_edges)


def sample_neighbors_in_score_graph(note_array, depth, samples_per_node, targets):
	"""
	Python wrapper function for C Extension function c_sample_neighbors_in_score_graph
	Samples Neighbors within a score graph
	In comparison to other methods involving pre-neighbors, this one doesn't use a lookup table for the neighborhood of a node,
	but it computes the neighborhood of a node on the fly which can be done efficiently due to the form that neighborhoods have in score graphs

	Parameters
	----------
	note_array : partitura/numpy.structured array
		This represents a score graph, as in, the data in this structure determines the edges between nodes
		required fields: onset_div, duration_div
		note_array['onset_div'] is a non-decreasing integer array
		note_array['duration_div'] is an integer array
	depth : int
		The number of layers that are sampled
	samples_per_node : int
		The number of samples per node.
	targets : np.ndarray
		initial value for the sampling iteration

	Note: c_sample_neighbors_in_score_graph expects onsets and durations to be passed in as int32 integer arrays (see code below)

	Returns
	-------
	samples_per_layer: PyList(type=np.ndarray, length=depth+1)
		List of numpy arrays of nodes (called layers) where the last layer corresponds to 'targets' and each n-th layer which isn't the last is a subset of the pre-neighborhood of the n+1-th layer
	edges_between_layers: PyList(type=np.ndarray(2, N), length=depth)
		List of numpy arrays of edges which show how 2 consecutive layers in samples_per_layer are connected 
	"""
	assert len(targets)>0

	onsets = note_array["onset_div"].astype(numpy.int32)
	durations = note_array["duration_div"].astype(numpy.int32)
	samples_per_layer, edges_between_layers = c_sample_neighbors_in_score_graph(onsets, durations, depth, samples_per_node, targets)
	# move to torch tensors
	samples_per_layer = [torch.from_numpy(layer) for layer in samples_per_layer]
	edges_between_layers = [torch.from_numpy(edges) for edges in edges_between_layers]
	return samples_per_layer, edges_between_layers


def sample_preneighbors_within_region(cgraph, region, samples_per_node=10):
	"""
	Python wrapper function for C Extension function c_sample_preneighbors_within_region
	Samples the pre-neighbors (or predecessors) within a score region.

	Parameters
	----------
	cgraph : Graph
		The score graph implemented in c. It is an attribute of the HeteroScoreGraph.
	region : tuple
		The region to sample from. It is a tuple of two integers, start and end.
		The region is inclusive on the left and exclusive on the right.
	samples_per_node : int
		The number of samples per node.

	Note: c_sample_preneighbors_within_region expects the region to be passed as 2 separate integers (see return statement)

	Returns
	-------
	Samples: np.ndarray
		The sampled nodes. It is a 1D array of integers. It might not contain all nodes in the region.
	edges: np.ndarray (2, num_edges)
		The edges. It is a 2D array of integers. The first row contains the source nodes, the second row the destination nodes.
	"""
	region_start, region_end = region

	if region_start>=region_end:
		raise ValueError("invalid region given")
	samples, edges = c_sample_preneighbors_within_region(cgraph, region_start, region_end, samples_per_node)
	# move to torch tensors
	samples = torch.from_numpy(samples)
	edges = torch.from_numpy(edges)
	return samples, edges