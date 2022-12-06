# import sys

# sys.settrace()

import samplers as gmsamplers

# edge_list = [
# 	(gmsamplers.Node(0), gmsamplers.Node(1)),
# 	#(gmsamplers.Node(0), gmsamplers.Node(9)),
# 	(gmsamplers.Node(1), gmsamplers.Node(2)),
# 	#(gmsamplers.Node(1), gmsamplers.Node(8)),
# 	(gmsamplers.Node(2), gmsamplers.Node(3)),
# 	#(gmsamplers.Node(2), gmsamplers.Node(7)),
# 	#(gmsamplers.Node(3), gmsamplers.Node(4)),
# 	(gmsamplers.Node(3), gmsamplers.Node(6)),
# 	#(gmsamplers.Node(4), gmsamplers.Node(5)),
# 	(gmsamplers.Node(5), gmsamplers.Node(4)),
# 	(gmsamplers.Node(5), gmsamplers.Node(6)),
# 	#(gmsamplers.Node(6), gmsamplers.Node(7)),
# 	(gmsamplers.Node(6), gmsamplers.Node(3)),
# 	#(gmsamplers.Node(7), gmsamplers.Node(8)),
# 	#(gmsamplers.Node(7), gmsamplers.Node(2)),
# 	(gmsamplers.Node(8), gmsamplers.Node(9)),
# 	(gmsamplers.Node(8), gmsamplers.Node(1)),
# 	#(gmsamplers.Node(9), gmsamplers.Node(0)),
# 	(gmsamplers.Node(0), gmsamplers.Node(0)),
# 	#(gmsamplers.Node(0), gmsamplers.Node(8)),
# 	(gmsamplers.Node(1), gmsamplers.Node(1)),
# 	(gmsamplers.Node(1), gmsamplers.Node(7)),
# 	#(gmsamplers.Node(2), gmsamplers.Node(2)),
# 	(gmsamplers.Node(2), gmsamplers.Node(6)),
# 	#(gmsamplers.Node(3), gmsamplers.Node(3)),
# 	(gmsamplers.Node(3), gmsamplers.Node(5)),
# 	(gmsamplers.Node(4), gmsamplers.Node(4)),
# 	#(gmsamplers.Node(5), gmsamplers.Node(3)),
# 	(gmsamplers.Node(5), gmsamplers.Node(5)),
# 	#(gmsamplers.Node(6), gmsamplers.Node(6)),
# 	(gmsamplers.Node(6), gmsamplers.Node(2)),
# 	(gmsamplers.Node(7), gmsamplers.Node(7)),
# 	#(gmsamplers.Node(7), gmsamplers.Node(1)),
# 	#(gmsamplers.Node(8), gmsamplers.Node(8)),
# 	(gmsamplers.Node(8), gmsamplers.Node(0)),
# 	#(gmsamplers.Node(9), gmsamplers.Node(9))
# ]


# edge_list.sort(key=lambda t: t[0].index())

edge_list = [
	(0, 1),
	#(0), 9)),
	(1, 2),
	#(1), 8)),
	(2, 3),
	#(2), 7)),
	#(3), 4)),
	(3, 6),
	#(4), 5)),
	(5, 4),
	(5, 6),
	#(6), 7)),
	(6, 3),
	#(7), 8)),
	#(7), 2)),
	(8, 9),
	(8, 1),
	#(9), 0)),
	(0, 0),
	#(0), 8)),
	(1, 1),
	(1, 7),
	#(2), 2)),
	(2, 6),
	#(3), 3)),
	(3, 5),
	(4, 4),
	#(5), 3)),
	(5, 5),
	#(6), 6)),
	(6, 2),
	(7, 7),
	#(7), 1)),
	#(8), 8)),
	(8, 0),
	#(9), 9))
]

edge_list.sort(key=lambda t: t[0])

graph = gmsamplers.Graph(edge_list)

graph.print()

target=[1, 9]

samples_per_layer, edge_indices_between_layers, load_per_layer = gmsamplers.sample_neighbors(graph, 5, 2, target)



for l in range(len(edge_indices_between_layers)):
	edge_indices = edge_indices_between_layers[l]

	# edges = [[edge_list[i][0].index(), edge_list[i][1].index()] for i in edge_indices if i is not None]

	# dst_nodes = [n.index() for n in samples_per_layer[l] if n]
	# src_nodes = [n.index() for n in samples_per_layer[l+1] if n]

	edges = [[edge_list[i][0], edge_list[i][1]] for i in edge_indices if i is not None]

	dst_nodes = [n for n in samples_per_layer[l] if n]
	src_nodes = [n for n in samples_per_layer[l+1] if n]

	print(dst_nodes)
	print(edges)
	print(src_nodes)
	#print([n.index() for n in load_per_layer[l]])
	print([n for n in load_per_layer[l]])
	print("--------------------------")

	# for e1,e2 in zip(zip(src_nodes, dst_nodes), edges):
	# 	assert(e1==e2), f"{e1}!={e2}"

	



# print([n.index() if n else '*' for n in samples_per_layer[0]])

# for l in range(len(edge_indices_between_layers)):
# 	print(edge_list[edge_indices_between_layers[l]]);
# 	print([n.index() if n else '*' for n in samples_per_layer[l+1]])