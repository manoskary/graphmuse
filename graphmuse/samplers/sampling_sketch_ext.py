import numpy

def extend_score_region_via_neighbor_sampling(graph, onset, duration, region_start, region_end, samples_per_node):
	samples=set()
	endtimes = onset + duration
	endtimes_cummax = numpy.maximum.accumulate(endtimes)

	onset_ref = None

	if region_start>0:
		for j in range(region_start, region_end):
			''' early exit strategy:
					if the note j has an onset larger than the maximum endtime from the notes [0,region_start[
					and j's onset isn't the closest to the maximum endtime,
					then j has no possible pre-neighbor in the region [0,region_start[.
					But since onset is increasing, all notes above j have no pre-neighbors in [0,region_start[ either
					therefore the loop can be exited early
			''' 
			if onset[j]>= endtimes_cummax[region_start-1]:
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

			for i in numpy.random.choice(pre_n[:marker], min(marker,samples_per_node), replace=False):
				samples.add(i)

	if region_end<=len(onset)-1:
		for i in range(region_end-1, region_start-1, -1):
			if endtimes_cummax[i]<=onset[region_end-1]:
				break

			neighbors_i = []
			for j in range(region_end, len(onset)):
				if onset[i]+duration[i]>onset[j]:
					neighbors_i.append(j)

				if onset[i]+duration[i]==onset[j]:
					neighbors_i.append(j)
					break

				if onset[i]+duration[i]<onset[j]:
					neighbors_i.append(j)
					break

			for j in numpy.random.choice(neighbors_i, min(len(neighbors_i), samples_per_node), replace=False):
				samples.add(j)

	return samples