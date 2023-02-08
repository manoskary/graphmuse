from .csamplers import *
import numpy

def graph(edges):
	if edges.dtype != numpy.uint32:
		raise TypeError("currently only numpy.uint32 nodes supported")

	node_count = max(numpy.max(edges[0]), edges[1][-1])+1

	# TODO: change strides such that src and dst can be iterated separately without strides in C code

	return Graph(edges, node_count)