import numpy






def random_score_pre_neighbor(j, note_array):
	while True:
		i = numpy.random.randint(0,j,1)

		if note_array['onset_div'][i] == note_array['onset_div'][j]:
			return i

		end_time_i = note_array['onset_div'][i]+note_array['duration_div'][i]

		if end_time_i == note_array['onset_div'][j]:
			return i

		if end_time_i > note_array['onset_div'][j]:
			return i

		for k in range(i+1,j):
			if note_array['onset_div'][k] == end_time_i:
				break

			if note_array['onset_div'][k] > end_time_i:
				if note_array['onset_div'][k] == note_array['onset_div'][j]:
					return i

				break


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
		assert False, "Impossible, something is wrong with the code"
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
			(samples_start,samples_end)=(0,graphs[g_idx].size())
		else:
			(samples_start,samples_end)=random_score_region(graphs[g_idx].note_array['onset_div'], max_subgraph_size, check_possibility)
			assert samples_end-samples_start<=max_subgraph_size

		subgraphs.append((g_idx,(samples_start,samples_end)))

	return subgraphs



note_arrays = [sorted(numpy.random.randint(0,20,size=numpy.random.randint(1,20))) for _ in range(10)]

for i,n in enumerate(note_arrays):
	print(i,":",n)
print("-------------------------------------------------")
graphs = [Graph({'onset_div':n}) for n in note_arrays]

y=musical_sampling(graphs, 10, 7)

for g_idx,(l,r) in y:
	print(g_idx,":",l, r, note_arrays[g_idx][l:r])