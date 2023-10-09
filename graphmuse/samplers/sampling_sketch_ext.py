import numpy

def extend_score_region_via_neighbor_sampling(graph, onset, duration, region_start, region_end, samples_per_node):
	samples=set()
	endtimes = onset + duration
	endtimes_cummax = numpy.maximum.accumulate(endtimes)

	onset_ref = None

	for j in range(region_start, region_end):
		''' early exit strategy:
				if the note j has an onset larger than the maximum endtime from the notes [0,region_start[
				and j's onset isn't the closest to the maximum endtime,
				then j has no possible pre-neighbor in the region [0,region_start[.
				But since onset is increasing, all notes above j have no pre-neighbors in [0,region_start[ either
				therefore the loop can be exited early
		''' 
		if region_start>0 and onset[j]>endtimes_cummax[region_start-1]:
			if onset_ref:
				# as long as the onsets remain constant, we continue the loop
				# only when it changes, we exit
				if onset_ref!=onset[j]:
					break
			else:
				onset_ref=onset[j]

		# ASSUMPTION: pre_neighbors are sorted
		pre_n = graph.pre_neighbors(j)

		# get the boundary where pre-neighbors are in [0,region_start[
		marker=0
		while marker<len(pre_n) and pre_n[marker]<region_start:
			marker+=1

		# sample min(marker, samples_per_node) pre-neighbors
		if marker<samples_per_node:
			for i in pre_n[:marker]:
				samples.add(i)
		else:
			perm = numpy.random.permutation(pre_n[:marker])
			for i in perm[:samples_per_node]:
				samples.add(i)

	return samples