static PyObject* random_score_region(PyObject* csamplers, PyObject* args){
	PyArrayObject* np_onsets;
	PyArrayObject* np_unique_onset_indices;
	Index budget;
	#define uint unsigned int

	if(!PyArg_ParseTuple(args, "OOI", (PyObject**)&np_onsets, (PyObject**)&np_unique_onset_indices, (uint*)&budget)){
		printf("If you don't provide proper arguments, you can't get a random score region.\nHow can you get a random score region if you don't provide proper arguments?\n");
		return NULL;
	}

	int* onsets = (int*)PyArray_DATA(np_onsets);
	Index* unique_onset_indices = (Index*)PyArray_DATA(np_unique_onset_indices);

	Index perm_size = (Index)PyArray_SIZE(np_unique_onset_indices);

	Index* perm = (Index*)malloc(sizeof(Index)*perm_size);
	for(Index i=0; i<perm_size; i++)
		perm[i]=i;

	Index region_start=0, region_end=0;

	for(Index i=0; i<perm_size; i++){
		Index rand_i = i + rand()%(perm_size-i);

		region_start = unique_onset_indices[perm[rand_i]];

		if(region_start + budget >= PyArray_SIZE(np_onsets)){
			region_end = (Index)PyArray_SIZE(np_onsets);
			break;
		}

		region_end = region_start+budget;

		while(region_end-1>=region_start && onsets[region_end]==onsets[region_end-1]){
			ASSERT(region_end>=1);

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
	int sample_rightmost;

	if(!PyArg_ParseTuple(args, "OOOOIIIp", (PyObject**)&graph, (PyObject**)&np_onsets, (PyObject**)&np_durations, (PyObject**)&np_endtimes_cummax, (uint*)&region_start, (uint*)&region_end, (uint*)&samples_per_node, &sample_rightmost)){
		printf("If you don't provide proper arguments, you can't extend a score region via neighbor sampling.\nHow can you extend a score region via neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	HashSet samples, node_tracker;
	HashSet_new(&samples, region_end-region_start);
	HashSet_new(&node_tracker, samples_per_node);
	

	PyArrayObject* left_extension;
	PyArrayObject* left_edges;
	PyArrayObject* right_extension;
	PyArrayObject* right_edges;

	Index edge_list_size = region_end-region_start;
	Node* edge_list = (Node*)malloc(3*sizeof(Node)*edge_list_size);
	ASSERT(edge_list);
	Index edge_list_cursor;
	

	int* onsets = (int*)PyArray_DATA(np_onsets);
	int* durations = (int*)PyArray_DATA(np_durations);
	int* endtimes_cummax = (int*)PyArray_DATA(np_endtimes_cummax);

	if(region_start > 0){
		int onset_ref = -1;
		HashSet_init(&samples);
		edge_list_cursor = 0;

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

			Index predict_cursor = edge_list_cursor + MACRO_MIN(marker, samples_per_node);

			if(predict_cursor  >= edge_list_size){
				Index new_size = (Index)(edge_list_size*1.5f);

				while(predict_cursor>=new_size)
					new_size = (Index)(new_size*1.5f);

				Node* tmp = edge_list;
				edge_list = (Node*)malloc(3*sizeof(Node)*new_size);

				ASSERT(edge_list);

				memcpy(edge_list, tmp, 3*sizeof(Node)*edge_list_size);

				free(tmp);
				edge_list_size = new_size;
			}

			if(marker <= samples_per_node){
				for(Index i=0; i<marker; i++){
					Node pre_neighbor = src_node_at(graph, offset + i);
					HashSet_add_node(&samples, pre_neighbor);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=pre_neighbor;
					edge_list[3*edge_list_cursor+1]=Index_To_Node(j);
					edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + i);
					edge_list_cursor++;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (Index)(0.75*marker)){
				Index* perm = (Index*)malloc(sizeof(Index)*marker);

				ASSERT(perm);

				for(Index i=0; i<marker; i++)
					perm[i]=i;
				

				for(Index i=0; i<samples_per_node; i++){
					Index rand_i = i + rand()%(marker-i);

					Node pre_neighbor = src_node_at(graph, offset + perm[rand_i]);

					HashSet_add_node(&samples, pre_neighbor);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=pre_neighbor;
					edge_list[3*edge_list_cursor+1]=Index_To_Node(j);
					edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + perm[rand_i]);
					edge_list_cursor++;

					perm[rand_i]=perm[i];
				}

				free(perm);
			}
			else{
				HashSet_init(&node_tracker);

				for(Index sample=0; sample<samples_per_node; sample++){
					Index edge_index;

					Node pre_neighbor;

					for(;;){
						edge_index = rand()%marker;
						pre_neighbor = src_node_at(graph, offset + edge_index);
						if(HashSet_add_node(&node_tracker, pre_neighbor))
							break;
					}

					HashSet_add_node(&samples, pre_neighbor);

					ASSERT(edge_list_cursor < edge_list_size);
					
					edge_list[3*edge_list_cursor]=pre_neighbor;
					edge_list[3*edge_list_cursor+1]=Index_To_Node(j);
					edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + edge_index);
					edge_list_cursor++;
				}
			}
		}
		left_extension = HashSet_to_numpy(&samples);
		left_edges = numpy_edge_list(edge_list, edge_list_cursor);	
	}
	else{
		left_extension = new_node_numpy(0);
		left_edges = numpy_edge_list(NULL, 0);	
	}


	if(sample_rightmost && region_end<=PyArray_SIZE(np_onsets)-1){
		HashSet_init(&samples);
		edge_list_cursor = 0;

		for(Int i=region_end-1; i>=(Int)region_start; i--){
			if(endtimes_cummax[i] <= onsets[region_end-1])
				break;

			Index marker = region_end;

			while(marker < PyArray_SIZE(np_onsets) && onsets[marker] <= onsets[i]+durations[i])
				marker++;

			if(((marker < PyArray_SIZE(np_onsets)) & (marker >= 1)) && onsets[marker-1] < onsets[i]+durations[i])
				marker++;

			Index predict_cursor = edge_list_cursor + MACRO_MIN(marker-region_end, samples_per_node);

			if(predict_cursor  >= edge_list_size){
				Index new_size = (Index)(edge_list_size*1.5f);

				while(predict_cursor>=new_size)
					new_size = (Index)(new_size*1.5f);

				Node* tmp = edge_list;
				edge_list = (Node*)malloc(3*sizeof(Node)*new_size);

				ASSERT(edge_list);

				memcpy(edge_list, tmp, 3*sizeof(Node)*edge_list_size);

				free(tmp);
				edge_list_size = new_size;
			}

			if(marker-region_end <= samples_per_node){
				for(Node j=region_end; j<marker; j++){
					HashSet_add_node(&samples, j);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=Index_To_Node(j);
					edge_list[3*edge_list_cursor+1]=Index_To_Node((Index)i);

					// TODO: edge_list[3*edge_list_cursor+2]= what type?

					edge_list_cursor++;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (Index)(0.75*(marker-region_end))){
				Node* perm = (Node*)malloc(sizeof(Node)*(marker-region_end));

				ASSERT(perm);

				for(Index j=0; j<marker-region_end; j++)
					perm[j]=Index_To_Node(j+region_end);
				

				for(Index j=0; j<samples_per_node; j++){
					Index rand_j = j + rand()%(marker-region_end-j);

					Node node_sample = perm[rand_j];

					HashSet_add_node(&samples, node_sample);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=node_sample;
					edge_list[3*edge_list_cursor+1]=Index_To_Node((Index)i);

					// TODO: edge_list[3*edge_list_cursor+2] = what type?

					//	onsets[node_sample]==onsets[i] => edge_list[3*edge_list_cursor+2] = Onset
					//  onsets[node_sample]==onsets[i]+durations[i] => edge_list[3*edge_list_cursor+2] = Consecutive
					// etc.


					edge_list_cursor++;

					perm[rand_j]=perm[j];
				}

				free(perm);
			}
			else{ //rejection sampling
				HashSet_init(&node_tracker);

				for(Index sample=0; sample<samples_per_node; sample++){
					Node node_sample;

					for(;;){
						node_sample = region_end + rand()%(marker-region_end);
						if(HashSet_add_node(&node_tracker, node_sample))
							break;
					}

					HashSet_add_node(&samples, node_sample);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=node_sample;
					edge_list[3*edge_list_cursor+1]=Index_To_Node((Index)i);

					// TODO: edge_list[3*edge_list_cursor+2]= what type based on (node_sample, i)?



					edge_list_cursor++;
				}
			}
		}
		right_extension = HashSet_to_numpy(&samples);
		right_edges = numpy_edge_list(edge_list, edge_list_cursor);	
	}
	else{
		right_extension = new_node_numpy(0);
		right_edges = numpy_edge_list(NULL, 0);	
	}
	
	free(edge_list);
	HashSet_free(&samples);
	HashSet_free(&node_tracker);
	
	return PyTuple_Pack(2, PyTuple_Pack(2, left_extension, left_edges), PyTuple_Pack(2, right_extension, right_edges));
}

static PyObject* sample_neighbors_in_score_graph(PyObject* csamplers, PyObject* args){
	PyArrayObject* np_onsets;
	PyArrayObject* np_durations;

	Index depth;
	Index samples_per_node;

	PyArrayObject* targets;


	if(!PyArg_ParseTuple(args, "OOIIO", (PyObject**)&np_onsets, (PyObject**)&np_durations, (uint*)&depth, (uint*)&samples_per_node, (PyObject**)&targets)){
		printf("If you don't provide proper arguments, you can't sample neighbors in a score graph.\nHow can you sample neighbors in a score graph if you don't provide proper arguments?\n");
		return NULL;
	}	

	PyObject* samples_per_layer = PyList_New(depth+1);
	PyObject* edges_between_layers = PyList_New(depth);

	if((samples_per_layer == NULL) | (edges_between_layers == NULL)){
		printf("can't create return pylists\n");

		Py_XDECREF(samples_per_layer);
		Py_XDECREF(edges_between_layers);

		return NULL;
	}

	PyList_SET_ITEM(samples_per_layer, depth, (PyObject*)targets);

	PyArrayObject* prev_layer = targets;

	int* onsets = (int*)PyArray_DATA(np_onsets);
	int* durations = (int*)PyArray_DATA(np_durations);

	HashSet node_hash_set;
	HashSet_new(&node_hash_set, (Index)PyArray_SIZE(prev_layer));

	HashSet total_samples;
	HashSet_new(&total_samples, (Index)PyArray_SIZE(prev_layer));
	HashSet_init(&total_samples);

	HashSet node_tracker;
	HashSet_new(&node_tracker, samples_per_node);

	Index edge_list_size = (Index)PyArray_SIZE(prev_layer)*(Index)power(samples_per_node, depth);
	Node* edge_list = (Node*)malloc(3*sizeof(Node)*edge_list_size);

	ASSERT(edge_list);


	
	for(uint layer=depth;layer>0; layer--){
		Index edge_list_cursor=0;

		HashSet_init(&node_hash_set);

		Node* prev_layer_nodes = (Node*)PyArray_DATA(prev_layer);

		Index prev_size = (Index)PyArray_SIZE(prev_layer);
		for(Index n=0; n<prev_size; n++){
			Node src_node = prev_layer_nodes[n];

			Index i = Node_To_Index(src_node);

			Index lower_bound, upper_bound;
			if(i == 0){
				lower_bound = 0;
			}
			else{
				lower_bound = i-1;

				while(lower_bound > 0 && onsets[lower_bound]==onsets[i])
					lower_bound--;

				if(lower_bound > 0)
					lower_bound++;
				else if(onsets[0]!=onsets[i])
					lower_bound = 1;
			}

			if(i == PyArray_SIZE(np_onsets)-1){
				upper_bound = (Index)PyArray_SIZE(np_onsets);
			}
			else{
				upper_bound = i+1;

				while(upper_bound < PyArray_SIZE(np_onsets) && onsets[upper_bound] < onsets[i] + durations[i])
					upper_bound++;

				if(upper_bound < PyArray_SIZE(np_onsets))
					upper_bound++;
			}
			
			Index neighbor_count = upper_bound-lower_bound;

			if(neighbor_count <= samples_per_node){
				for(Index j=lower_bound; j<upper_bound; j++){
					HashSet_add_node(&node_hash_set, j);
					HashSet_add_node(&total_samples, j);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=j;
					edge_list[3*edge_list_cursor+1]=i;
					edge_list_cursor++;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (Index)(0.75*neighbor_count)){
				Node* perm = (Node*)malloc(sizeof(Node)*neighbor_count);

				ASSERT(perm);

				for(Index ix=0; ix<neighbor_count; ix++)
					perm[ix]=Index_To_Node(ix+lower_bound);
				

				for(Index ix=0; ix<samples_per_node; ix++){
					Index rand_ix = ix + rand()%(neighbor_count-ix);

					Node j = perm[rand_ix];

					HashSet_add_node(&node_hash_set, j);
					HashSet_add_node(&total_samples, j);

					ASSERT(edge_list_cursor < edge_list_size);

					edge_list[3*edge_list_cursor]=j;
					edge_list[3*edge_list_cursor+1]=i;
					edge_list_cursor++;

					perm[rand_ix]=perm[ix];
				}

				free(perm);
			}
			else{
				HashSet_init(&node_tracker);

				for(Index sample=0; sample<samples_per_node; sample++){
					Node j;

					for(;;){
						j = lower_bound + rand()%neighbor_count;	
						if(HashSet_add_node(&node_tracker, j))
							break;
					}

					HashSet_add_node(&node_hash_set, j);
					HashSet_add_node(&total_samples, j);

					ASSERT(edge_list_cursor < edge_list_size);
					

					edge_list[3*edge_list_cursor]=j;
					edge_list[3*edge_list_cursor+1]=i;
					edge_list_cursor++;
				}
			}
		}


		
		PyArrayObject* edges = numpy_edge_list(edge_list, edge_list_cursor);
		

		PyArrayObject* new_layer = HashSet_to_numpy(&node_hash_set);

		prev_layer = new_layer;

		PyList_SET_ITEM(samples_per_layer, layer-1, (PyObject*)new_layer);
		PyList_SET_ITEM(edges_between_layers, layer-1, (PyObject*)edges);
	}

	PyArrayObject* np_total_samples = HashSet_to_numpy(&total_samples);

	HashSet_free(&total_samples);
	HashSet_free(&node_tracker);
	HashSet_free(&node_hash_set);
	free(edge_list);

	return PyTuple_Pack(3, samples_per_layer, edges_between_layers, np_total_samples);
}

static PyObject* sample_preneighbors_within_region(PyObject* csamplers, PyObject* args){
	Graph* graph;
	Index region_start;
	Index region_end;
	Index samples_per_node;

	if(!PyArg_ParseTuple(args, "OIII", (PyObject**)&graph, (uint*)&region_start, (uint*)&region_end, (uint*)&samples_per_node)){
		printf("If you don't provide proper arguments, you can't extend a score region via neighbor sampling.\nHow can you extend a score region via neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	HashSet samples, node_tracker;
	HashSet_new(&samples, region_end-region_start);
	HashSet_init(&samples);
	HashSet_new(&node_tracker, samples_per_node);

	Index edge_list_size = samples_per_node*(region_end-region_start);
	Node* edge_list = (Node*)malloc(3*sizeof(Node)*edge_list_size);
	Index edge_list_cursor=0;

	ASSERT(edge_list);

	for(Index j=region_start+1; j<region_end; j++){
		Index offset = graph->pre_neighbor_offsets[j];
		Index pre_neighbor_count = graph->pre_neighbor_offsets[j+1]-offset;

		Index intersection_start = 0;

		while(intersection_start < pre_neighbor_count && Node_To_Index(src_node_at(graph, offset+intersection_start)) < region_start)
			intersection_start++;

		Index intersection_end = intersection_start+1;

		while(intersection_end < pre_neighbor_count && Node_To_Index(src_node_at(graph, offset+intersection_end)) < region_end)
			intersection_end++;

		Index intersection_count = intersection_end - intersection_start;


		if(intersection_count <= samples_per_node){
			for(Index ix=intersection_start; ix < intersection_end; ix++){
				Node i = src_node_at(graph, offset + ix);
				HashSet_add_node(&samples, i);

				ASSERT(edge_list_cursor < edge_list_size);

				edge_list[3*edge_list_cursor]=i;
				edge_list[3*edge_list_cursor+1]=Index_To_Node(j);
				edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + ix);

				edge_list_cursor++;
			}
		}
		/*
			expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
			for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
			to let's say 4, f has to be at most 3/4=0.75

			if this threshold is reached, random subset is sampled via random permutation
			this is viable since memory waste is at most 25% (for temporary storage)
		*/
		else if(samples_per_node > (Index)(0.75*intersection_count)){
			Index* perm = (Index*)malloc(sizeof(Index)*intersection_count);

			ASSERT(perm);

			for(Index ix=0; ix<intersection_count; ix++)
				perm[ix]=ix;
			

			for(Index ix=0; ix<samples_per_node; ix++){
				Index rand_ix = ix + rand()%(intersection_count-ix);

				ASSERT(rand_ix < intersection_count);

				Node i = src_node_at(graph, offset + perm[rand_ix]);

				HashSet_add_node(&samples, i);

				ASSERT(edge_list_cursor < edge_list_size);

				edge_list[3*edge_list_cursor]=i;
				edge_list[3*edge_list_cursor+1]=Index_To_Node(j);
				edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + perm[rand_ix]);
				edge_list_cursor++;

				perm[rand_ix]=perm[ix];
			}

			free(perm);
		}
		else{
			HashSet_init(&node_tracker);

			for(Index sample=0; sample<samples_per_node; sample++){
				Node i;
				Index ix;

				for(;;){
					ix = intersection_start + rand()%intersection_count;
					i = src_node_at(graph, offset + ix)	;
					if(HashSet_add_node(&node_tracker, i))
						break;
				}

				HashSet_add_node(&samples, i);

				ASSERT(edge_list_cursor < edge_list_size);
				
				edge_list[3*edge_list_cursor]=i;
				edge_list[3*edge_list_cursor+1]=(Node)j;
				edge_list[3*edge_list_cursor+2]=edge_type_at(graph, offset + ix);
				edge_list_cursor++;
			}
		}
	}

	PyArrayObject* np_samples = HashSet_to_numpy(&samples);
	PyArrayObject* np_edges = numpy_edge_list(edge_list, edge_list_cursor);

	free(edge_list);
	HashSet_free(&samples);
	HashSet_free(&node_tracker);

	return PyTuple_Pack(2, np_samples, np_edges);
}