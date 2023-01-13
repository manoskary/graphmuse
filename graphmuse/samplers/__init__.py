from .csamplers import *
import numpy

def graph(edges):
	if edges.dtype != numpy.uint32:
		raise TypeError("currently only numpy.uint32 nodes supported")

	node_count = max(numpy.max(edges[1]), edges[0][-1])+1

	return Graph(edges, node_count)