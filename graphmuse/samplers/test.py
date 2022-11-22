# import sys

# sys.settrace()

import graphmuse

edge_list = [
	(graphmuse.Node(0), graphmuse.Node(1)),
	#(graphmuse.Node(0), graphmuse.Node(9)),
	(graphmuse.Node(1), graphmuse.Node(2)),
	#(graphmuse.Node(1), graphmuse.Node(8)),
	(graphmuse.Node(2), graphmuse.Node(3)),
	#(graphmuse.Node(2), graphmuse.Node(7)),
	#(graphmuse.Node(3), graphmuse.Node(4)),
	(graphmuse.Node(3), graphmuse.Node(6)),
	#(graphmuse.Node(4), graphmuse.Node(5)),
	(graphmuse.Node(5), graphmuse.Node(4)),
	(graphmuse.Node(5), graphmuse.Node(6)),
	#(graphmuse.Node(6), graphmuse.Node(7)),
	(graphmuse.Node(6), graphmuse.Node(3)),
	#(graphmuse.Node(7), graphmuse.Node(8)),
	#(graphmuse.Node(7), graphmuse.Node(2)),
	(graphmuse.Node(8), graphmuse.Node(9)),
	(graphmuse.Node(8), graphmuse.Node(1)),
	#(graphmuse.Node(9), graphmuse.Node(0)),
	(graphmuse.Node(0), graphmuse.Node(0)),
	#(graphmuse.Node(0), graphmuse.Node(8)),
	(graphmuse.Node(1), graphmuse.Node(1)),
	(graphmuse.Node(1), graphmuse.Node(7)),
	#(graphmuse.Node(2), graphmuse.Node(2)),
	(graphmuse.Node(2), graphmuse.Node(6)),
	#(graphmuse.Node(3), graphmuse.Node(3)),
	(graphmuse.Node(3), graphmuse.Node(5)),
	(graphmuse.Node(4), graphmuse.Node(4)),
	#(graphmuse.Node(5), graphmuse.Node(3)),
	(graphmuse.Node(5), graphmuse.Node(5)),
	#(graphmuse.Node(6), graphmuse.Node(6)),
	(graphmuse.Node(6), graphmuse.Node(2)),
	(graphmuse.Node(7), graphmuse.Node(7)),
	#(graphmuse.Node(7), graphmuse.Node(1)),
	#(graphmuse.Node(8), graphmuse.Node(8)),
	(graphmuse.Node(8), graphmuse.Node(0)),
	#(graphmuse.Node(9), graphmuse.Node(9))
]

edge_list.sort(key=lambda t: t[0].index())

graph = graphmuse.Graph(edge_list)

graph.print()

target=[graphmuse.Node(1), graphmuse.Node(9)]

samples_per_layer, edge_indices_between_layers, load_per_layer = graphmuse.sample_neighbors(graph, 5, 2, target)



for l in range(len(edge_indices_between_layers)):
	edge_indices = edge_indices_between_layers[l]

	edges = [[edge_list[i][0].index(), edge_list[i][1].index()] for i in edge_indices if i is not None]

	dst_nodes = [n.index() for n in samples_per_layer[l] if n]
	src_nodes = [n.index() for n in samples_per_layer[l+1] if n]

	print(dst_nodes)
	print(edges)
	print(src_nodes)
	print([n.index() for n in load_per_layer[l]])
	print("--------------------------")

	# for e1,e2 in zip(zip(src_nodes, dst_nodes), edges):
	# 	assert(e1==e2), f"{e1}!={e2}"

	



# print([n.index() if n else '*' for n in samples_per_layer[0]])

# for l in range(len(edge_indices_between_layers)):
# 	print(edge_list[edge_indices_between_layers[l]]);
# 	print([n.index() if n else '*' for n in samples_per_layer[l+1]])