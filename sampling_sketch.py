import numpy


def random_subrange(onsets, budget, check_possibility=True):
	if check_possibility:
		possible = False

		eq_count = 1

		cursor=0

		while cursor + eq_count < len(onsets):
			if onsets[cursor+eq_count]==onsets[cursor]:
				eq_count+=1
			elif eq_count<=budget:
				possible = True
				break
			else:
				cursor = cursor+eq_count
				eq_count = 1

		if not possible:
			raise ValueError("by including all notes with the same onset, the budget is always exceeded")

	while True:
		start_index = numpy.random.choice(len(onsets))

		l = start_index-1

		while l>=0 and onsets[l]==onsets[start_index]:
			l-=1

		l += 1

		def g(onsets, start, budget):
			cursor= start+1

			while cursor < len(onsets) and onsets[start]==onsets[cursor]:
				if cursor-start+1>budget:
					return start

				cursor+=1

			return cursor

		r = g(onsets, l, budget)



		if r==l:
			continue

		while r<len(onsets):
			rr = g(onsets, r, budget - (r-l))

			if rr==r:
				break

			r = rr

		return (l,r)



class Graph:
	def __init__(self,note_array):
		self.note_array = note_array

	def size(self):
		return len(self.note_array['onset_div'])



def musical_sampling(graphs, max_subgraph_size, subgraph_count, check_possibility=True):
	subgraphs = []
	
	graph_probs = numpy.empty(len(graphs))

	total_size = 0

	for i,g in enumerate(graphs):
		graph_probs[i] = g.size()
		total_size += graph_probs[i]

	graph_probs/=total_size

	for _ in range(subgraph_count):
		g_idx = numpy.random.choice(len(graphs), p=graph_probs)

		if graphs[g_idx].size()<=max_subgraph_size:
			(l,r)=(0,graphs[g_idx].size())
		else:
			(l,r)=random_subrange(graphs[g_idx].note_array['onset_div'], max_subgraph_size, check_possibility)
			assert r-l<=max_subgraph_size

		subgraphs.append((g_idx,(l,r)))

	return subgraphs



note_arrays = [sorted(numpy.random.randint(0,20,size=numpy.random.randint(1,20))) for _ in range(10)]

for n in note_arrays:
	print(n)
print("-------------------------------------------------")
graphs = [Graph({'onset_div':n}) for n in note_arrays]

y=musical_sampling(graphs, 10, 7)

for g_idx,(l,r) in y:
	print(g_idx,":",l, r)