import numpy


def random_score_region(onsets, budget, check_possibility=True):
	_, indices = numpy.unique(onsets,return_index=True)

	# in order to avoid handling the special case where a region is sampled that reaches to the end of 'onsets', we simply extend the possible values
	indices = numpy.concatenate([indices,[len(onsets)]])

	if check_possibility:
		if (numpy.diff(indices)>budget).all():
			raise ValueError("by including all notes with the same onset, the budget is always exceeded")

	# since we added the last element ourselves and it isn't a valid index,
	# we only sample excluding the last element
	# using a random permutation isn't necessarily, it just avoids sampling a previous sample
	for idx in numpy.random.permutation(len(indices)-1):
		samples_start = indices[idx]
		
		if samples_start+budget>=len(onsets):
			return (samples_start,len(onsets))

		samples_end = samples_start+budget

		while samples_end-1>=samples_start and onsets[samples_end]==onsets[samples_end-1]:
			samples_end-=1

		if samples_start<samples_end:
			return (samples_start, samples_end)


	if check_possibility:
		assert False, "a result should be possible, according to the check above, however, no result exists."
	else:
		raise ValueError("by including all notes with the same onset, the budget is always exceeded")




class Graph:
	def __init__(self,note_array):
		self.note_array = note_array

	def size(self):
		return len(self.note_array['onset_div'])



def musical_sampling(graphs, max_subgraph_size, subgraph_count, check_possibility=True):
	# we want to sample from the array 'graphs' proportional to the size of the graphs in the array
	# so we need to pre-compute a probability distribution for that
	graph_probs = numpy.empty(len(graphs))

	total_size = 0

	for i,g in enumerate(graphs):
		graph_probs[i] = g.size()
		total_size += graph_probs[i]

	graph_probs/=total_size

	# main loop
	subgraphs = []

	for _ in range(subgraph_count):
		g_idx = numpy.random.choice(len(graphs), p=graph_probs)

		if graphs[g_idx].size()<=max_subgraph_size:
			(l,r)=(0,graphs[g_idx].size())
		else:
			(l,r)=random_score_region(graphs[g_idx].note_array['onset_div'], max_subgraph_size, check_possibility)
			assert r-l<=max_subgraph_size

		subgraphs.append((g_idx,(l,r)))

	return subgraphs



note_arrays = [sorted(numpy.random.randint(0,20,size=numpy.random.randint(1,20))) for _ in range(10)]

for i,n in enumerate(note_arrays):
	print(i,":",n)
print("-------------------------------------------------")
graphs = [Graph({'onset_div':n}) for n in note_arrays]

y=musical_sampling(graphs, 10, 7)

for g_idx,(l,r) in y:
	print(g_idx,":",l, r, note_arrays[g_idx][l:r])