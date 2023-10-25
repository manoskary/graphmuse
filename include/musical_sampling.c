static PyObject* random_score_region(PyObject* csamplers, PyObject* args){
	PyArrayObject* np_onsets;
	PyArrayObject* np_unique_onset_indices;
	Index budget;

	if(!PyArg_ParseTuple(args, "OOI", (PyObject**)&np_onsets, (PyObject**)&np_unique_onset_indices, (uint*)&budget)){
		printf("If you don't provide proper arguments, you can't get a random score region.\nHow can you get a random score region if you don't provide proper arguments?\n");
		return NULL;
	}

	int* onsets = (int*)PyArray_DATA(np_onsets);
	Index* unique_onset_indices = (Index*)PyArray_DATA(np_unique_onset_indices);

	Index perm_size = PyArray_SIZE(np_unique_onset_indices);

	Index* perm = (Index*)malloc(sizeof(Index)*perm_size);
	for(Index i=0; i<perm_size; i++)
		perm[i]=i;

	Index region_start=0, region_end=0;

	for(Index i=0; i<perm_size; i++){
		Index rand_i = i + rand()%(perm_size-i);

		region_start = unique_onset_indices[perm[rand_i]];

		if(region_start + budget >= PyArray_SIZE(np_onsets)){
			region_end = PyArray_SIZE(np_onsets);
			break;
		}

		region_end = region_start+budget;

		while(region_end-1>=region_start && onsets[region_end]==onsets[region_end-1]){
			region_end--;
		}

		if(region_start < region_end)
			break;

		perm[rand_i] = perm[i];
	}

	free(perm);

	return PyTuple_Pack(2, PyLong_FromIndex(region_start), PyLong_FromIndex(region_end));
}


static PyObject* extend_score_region_via_neighbor_sampling(PyObject* csamplers, PyObject* args){
	Graph* graph;
	PyArrayObject* np_onsets;
	PyArrayObject* np_durations;
	PyArrayObject* np_endtimes_cummax;
	Index region_start;
	Index region_end;
	Index samples_per_node;

	if(!PyArg_ParseTuple(args, "OOOOIII", (PyObject**)&graph, (PyObject**)&np_onsets, (PyObject**)&np_durations, (PyObject**)&np_endtimes_cummax, (uint*)&region_start, (uint*)&region_end, (uint*)&samples_per_node)){
		printf("If you don't provide proper arguments, you can't extend a score region via neighbor sampling.\nHow can you extend a score region via neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	HashSet samples, node_tracker;
	HashSet_new(&samples, region_end-region_start);
	HashSet_new(&node_tracker, samples_per_node);
	HashSet_init(&samples);

	// Index edge_index_size = region_end-region_start;
	// Index* edge_index_canvas = (Index*)malloc(sizeof(Index)*edge_index_size);
	// ASSERT(edge_index_canvas);
	// Index edge_index_cursor = 0;
	

	int* onsets = (int*)PyArray_DATA(np_onsets);
	int* durations = (int*)PyArray_DATA(np_durations);
	int* endtimes_cummax = (int*)PyArray_DATA(np_endtimes_cummax);

	if(region_start > 0){
		int onset_ref = -1;

		for(Index j=region_start; j<region_end; j++){
			if(onsets[j] > endtimes_cummax[region_start-1]){
				if(onset_ref>=0){
					if(onset_ref != onsets[j])
						break;
				}
				else{
					onset_ref = onsets[j];
				}
			}

			Index offset = graph->pre_neighbor_offsets[j];
			Index pre_neighbor_count = graph->pre_neighbor_offsets[j+1]-graph->pre_neighbor_offsets[j];

			Index marker = 0;

			while(marker < pre_neighbor_count && Node_To_Index(src_node_at(graph, offset+marker)) < region_start)
				marker++;

			// Index predict_cursor = edge_index_cursor + MACRO_MIN(marker, samples_per_node);

			// if(predict_cursor  >= edge_index_size){
			// 	Index new_size = (Index)(edge_index_size*1.5f);

			// 	while(predict_cursor>=new_size)
			// 		new_size = (Index)(new_size*1.5f);

			// 	Index* tmp = edge_index_canvas;
			// 	edge_index_canvas = (Index*)malloc(sizeof(Index)*new_size);

			// 	ASSERT(edge_index_canvas);

			// 	memcpy(edge_index_canvas, tmp, sizeof(Index)*edge_index_size);

			// 	free(tmp);
			// 	edge_index_size = new_size;
			// }

			if(marker <= samples_per_node){
				for(Index i=0; i<marker; i++){
					Node pre_neighbor = src_node_at(graph, offset + i);
					HashSet_add_node(&samples, pre_neighbor);
					//edge_index_canvas[edge_index_cursor++] = offset + i;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (uint)(0.75*marker)){
				Index* perm = (Index*)malloc(sizeof(Index)*marker);

				ASSERT(perm);

				for(Index i=0; i<marker; i++)
					perm[i]=i;
				

				for(Index i=0; i<samples_per_node; i++){
					Index rand_i = i + rand()%(marker-i);

					Node node_sample = src_node_at(graph, offset + perm[rand_i]);

					HashSet_add_node(&samples, node_sample);

					//edge_index_canvas[edge_index_cursor++] = offset + perm[rand_i];

					perm[rand_i]=perm[i];
				}

				free(perm);
			}
			else{
				HashSet_init(&node_tracker);

				for(uint sample=0; sample<samples_per_node; sample++){
					Index edge_index;

					Node node_sample;

					for(;;){
						edge_index = rand()%marker;
						node_sample = src_node_at(graph, offset + edge_index);
						if(HashSet_add_node(&node_tracker, node_sample))
							break;
					}

					HashSet_add_node(&samples, node_sample);
					

					//edge_index_canvas[edge_index_cursor++] = offset + edge_index;
				}
			}
		}
	}

	if(region_end<=PyArray_SIZE(np_onsets)-1){
		for(Int i=region_end-1; i>=region_start; i--){
			if(endtimes_cummax[i] <= onsets[region_end-1])
				break;

			Index marker = region_end;

			while(marker < PyArray_SIZE(np_onsets) && onsets[marker] <= onsets[i]+durations[i])
				marker++;

			if(((marker < PyArray_SIZE(np_onsets)) & (marker >= 1)) && onsets[marker-1] < onsets[i]+durations[i])
				marker++;

			if(marker-region_end <= samples_per_node){
				for(Node j=region_end; j<marker; j++){
					HashSet_add_node(&samples, j);
					//edge_index_canvas[edge_index_cursor++] = offset + i;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (uint)(0.75*(marker-region_end))){
				Node* perm = (Node*)malloc(sizeof(Node)*(marker-region_end));

				ASSERT(perm);

				for(Index j=0; j<marker-region_end; j++)
					perm[j]=Index_To_Node(j+region_end);
				

				for(Index j=0; j<samples_per_node; j++){
					Index rand_j = j + rand()%(marker-region_end-j);

					Node node_sample = perm[rand_j];

					HashSet_add_node(&samples, node_sample);

					perm[rand_j]=perm[j];
				}

				free(perm);
			}
			else{
				HashSet_init(&node_tracker);

				for(uint sample=0; sample<samples_per_node; sample++){
					Node node_sample;

					for(;;){
						node_sample = region_end + rand()%(marker-region_end);
						if(HashSet_add_node(&node_tracker, node_sample))
							break;
					}

					HashSet_add_node(&samples, node_sample);
				}
			}
		}
	}
	

	//PyArrayObject* edge_indices = index_array_to_numpy(edge_index_canvas, edge_index_cursor);
		

	PyArrayObject* np_samples = HashSet_to_numpy(&samples);

	return (PyObject*)np_samples;//PyTuple_Pack(2, np_samples, edge_indices);
}
