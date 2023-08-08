import numpy
import graphmuse.samplers as gm_sampling
import time

for i in range(2,10):
	N = 10**(i//2+1)

	

	E = numpy.random.randint(1,max(N**2,2),1)[0]

	edges = numpy.random.randint(0,N,(2, E),dtype=numpy.uint32)

	edges = sorted(list(set([(edges[0,i], edges[1,i]) for i in range(E)])), key=lambda t:t[1])

	V = numpy.max(edges)



	edges = numpy.array(edges).T

	#dgl_graph = dgl.graph((torch.from_numpy(edges[0].astype(numpy.int32)),torch.from_numpy(edges[0].astype(numpy.int32))))

	gm_graph = gm_sampling.graph(edges)

	samples_per_node = numpy.random.randint(max(N//100,2),max(N//10,2),1)[0]

	print(f"N: {N}, E: {E}, V: {V}, S: {samples_per_node}")

	targets = numpy.random.choice(edges[1], min(samples_per_node,len(edges[1])), replace=False)

	t0 = time.perf_counter()

	_ = gm_sampling.sample_nodewise_mt_static(gm_graph, 2, samples_per_node, targets)

	t1 = time.perf_counter()

	#_ = dgl.sampling.sample_neighbors(dgl_graph, targets.astype(numpy.int32), samples_per_node)
	_ = gm_sampling.sample_nodewise(gm_graph, 2, samples_per_node, targets)

	t2 = time.perf_counter()

	print("MT vs ST:", t1-t0,t2-t1)