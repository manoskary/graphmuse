#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL sam_ARRAY_API
#include <ndarraytypes.h>
#include <ndarrayobject.h>



#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
#define MACRO_MIN(a,b) ((a)>=(b))? (b) : (a)

typedef uint32_t uint;
typedef uint32_t uint32;
typedef uint64_t uint64;

// these here are just used as indirection so that code can be compiled differently more easily
typedef int32_t Int;
typedef uint32_t Node;
typedef Node Index;
typedef Node Key;
typedef uint32_t Value;

#define Node_Eqv_In_Numpy NPY_UINT32
#define Index_Eqv_In_Numpy Node_Eqv_In_Numpy
#define Key_Eqv_In_Numpy Node_Eqv_In_Numpy
#define EType_Eqv_In_Numpy NPY_UINT8

typedef uint8_t EdgeType;
#define Onset_Eq (uint8_t)0
#define Onset_End_Eq (uint8_t)1
#define Inbetween (uint8_t)2
#define Closest_End (uint8_t)3

#define Node_To_Index(n) (n)
#define Index_To_Node(i) (i)


static PyArrayObject* new_node_numpy(Index size){
	const npy_intp dims = (npy_intp)size;
	return (PyArrayObject*)PyArray_SimpleNew(1, &dims, Node_Eqv_In_Numpy);
}



#include <hashmap.c>

// this should be fine actually since it is used on random nodes
static Key Node_hash(Node n){
	return (Key)n;
}

static bool HashSet_add_node(HashSet* hs, Node n){
	return HashSet_add(hs, Node_hash(n));
}

static PyArrayObject* HashSet_to_numpy(HashSet* hash_set){
	PyArrayObject* np_arr = new_node_numpy((Index)hash_set->size);

	Node* copy_dst = PyArray_DATA(np_arr);

	HashSet_copy(hash_set, copy_dst);	

	return np_arr;
}

/*
static void count_neighbors(Index from, Index to, Index node_count, Index* neighbor_counts, float* onset_beat, float* duration_beat, float* pitch){
	for(Index i=from; i<to; i++)
		for(Index j=0; j<node_count; j++)
			// need to account that neighbor_counts eventually gets accumulated to pre_neighbor_offsets which is 0 at index 0
			// hence i+1
			neighbor_counts[i+1]+=(Index)adjacency_map(i,j, onset_beat, duration_beat, pitch);
}

static void fill_in_neighbors(Index from, Index to, Index node_count, Index* pre_neighbor_offsets, PyArrayObject* edge_list, float* onset_beat, float* duration_beat, float* pitch){
	// to is excluded from range
	for(Index i=from; i<to; i++){
		Node* src = (Node*)PyArray_GETPTR2(edge_list, 0, pre_neighbor_offsets[i]);
		Node* dst = (Node*)PyArray_GETPTR2(edge_list, 1, pre_neighbor_offsets[i]);

		Index neighbor_count = pre_neighbor_offsets[i+1] - pre_neighbor_offsets[i];

		Index cursor=0;
		for(Index j=0; j<node_count; j++){
			src[cursor] = Index_To_Node(i);
			dst[cursor] = Index_To_Node(j);

			// cursor only moves if i and j are adjacent
			// meaning src[cursor] and dst[cursor] will be overwritten next iteration if i and j are not adjacent
			cursor += adjacency_map(i,j, onset_beat, duration_beat, pitch);

			if(cursor==neighbor_count)
				break;
		}
	}
}

*/

/*
static void** alloc_cacheline_apart(ssize_t required_count, ssize_t elem_size, uint thread_count){
	ssize_t required_bytes = elem_size*required_count;

	ssize_t req_bytes_per_thread = required_bytes/thread_count;

	ssize_t cache_line_size;

	ssize_t cache_lines_per_thread = req_bytes_per_thread/cache_line_size;

	if(req_bytes_per_thread%cache_line_size != 0)
		cache_lines_per_thread++;



	ssize_t total_bytes = required_bytes + (thread_count-1)*(cache_line_size*cache_lines_per_thread - req_bytes_per_thread)
	char* mem = (char*)malloc(total_bytes + thread_count*sizeof(void*));

	for(uint t=0; t < thread_count; t++){
		void* addr = (void*)(mem+sizeof(void*)*t);
		*addr = mem+sizeof(void*)*thread_count+cache_line_size*cache_lines_per_thread*t;
	}

	return (void**)mem;
}
*/




/*
static void count_pre_neighbors(Graph* graph, Index from, Index to, int* onset_div, int* duration_div, int max_duration_div){
	for(Index j=from; j<to; j++){
		Index pre_neighbor_count = 0;

		for(Index i=0; i<graph->node_count; i++){
			// can be localized
			pre_neighbor_count += (Index)(onset_div[i]==onset_div[j]);

			
			pre_neighbor_count += (Index)(onset_div[i] + duration_div[i] ==onset_div[j]);

			// can be localized
			pre_neighbor_count += (Index)((onset_div[i] < onset_div[j]) & (onset_div[j] < onset_div[i] + duration_div[i]));
			
			if(onset_div[j] > onset_div[i] + duration_div[i]){
				for(Index k=0; k<graph->node_count; k++){
					if(((onset_div[i] + duration_div[i] < onset_div[k]) & (onset_div[k] < onset_div[j])) | (onset_div[i] + duration_div[i] == onset_div[k]))
						goto OUTER_LOOP;
				}

				pre_neighbor_count++;
			}

		OUTER_LOOP:;
		}


		//	ASSUMPTION: the adjecancy checks below also apply to j itself
		// 	which means calloc is totally unnecessary
		// also, init value 1
		Index pre_neighbor_count = 1;

		//1.: count all onset_divs that are equal to onset_div[j] to the right of j
		for(Index i = j+1; i<graph->node_count; i++){
			if(onset_div[i] != onset_div[j])
				break;
			
			pre_neighbor_count++;
		}

		//2.: count all onset_divs that are equal to onset_div[j] to the left of j
		Int i = ((Int)j)-1;

		for(; i>=0; i--){
			if(onset_div[(Index)i] != onset_div[j])
				break;
			
			pre_neighbor_count++;
		}



		pre_neighbor_count += (Index)(duration_div[(Index)i] > 0);

		for(; i>=0; i--){
			if(onset_div[i] + max_duration_div < onset_div[j])
				break;

			pre_neighbor_count += (Index)(onset_div[(Index)i] + duration_div[(Index)i] >= onset_div[j]);
		}

		graph->pre_neighbor_offsets[i+1] = pre_neighbor_count;
	}
}



*/


//	TODO(?): make sure to minimize false sharing on neighbor_counts
// 	might require to create separate neighbor_counts for each thread, but those would have to lie a cache line apart in memory :/
//	Windows and Linux does provide some functionality for allocating memory at specific adresses (if memory serves well)

//	Note that threads are unlikely to write to same cache line since
//	overlapping cache lines only occur for last cache line for thread t and first cache line for thread t+1
//	but by the time thread t reaches last cache line, thread t+1 will most likely have moved on from first cache line
// 	This assumes though that every thread processes more than a cache line worth of data
//	(node_count/thread_count > cacheline_size/sizeof(Node)) is true for count_neighbors though if node_count is large enough

/* 	The reason for writing this code with 3 inner loops instead of looping over all pairs of (i,j),
	is exactly that, avoiding a quadratic runtime.
	By just looking at the local neighborhood of a note and exiting early if no adjecancy conditions longer hold,
	this should result on average in a sub-quadratic runtime
*/

static void count_neighbors(Index node_count, Index* neighbor_counts, Index from, Index to, int* onset_div, int* duration_div){
	for(Index i=from; i<to; i++){
		Index neighbor_count = 0;

		for(Int j=((Int)i)-1; j>=0; j--){
			bool
				cond1 = (onset_div[(Index)j] == onset_div[i]),
				cond3 = (onset_div[(Index)j] == onset_div[i] + duration_div[i]);

			neighbor_count += (Index)cond3 + (Index)cond1;

			if(! cond1)
				break;
		}



		Index j=i;

		for(;j<node_count; j++){
			bool
				cond1 = (onset_div[j] == onset_div[i]),
				cond3 = (onset_div[j] == onset_div[i] + duration_div[i]);

			neighbor_count += (Index)cond3 + (Index)cond1;

			if(! cond1)
				break;
		}

		Index j_star = j;

		for(;j<node_count; j++){
			bool
				cond2 = (onset_div[j] < onset_div[i] + duration_div[i]);

			neighbor_count+= (Index)cond2;

			if(! cond2)
				break;
		}

		if((j < node_count) && (onset_div[j] > onset_div[i] + duration_div[i])){
			int closest_from_above = onset_div[j];

			for(;j<node_count; j++){
				bool
					cond4 = (onset_div[j] == closest_from_above);

				neighbor_count+= (Index)cond4;

				if(! cond4)
					break;
			}
		}
		else{
			if(j == j_star)
				j++;
			for(;j<node_count; j++){
				bool
					cond1 = (onset_div[j] == onset_div[i]),
					cond3 = (onset_div[j] == onset_div[i] + duration_div[i]);

				neighbor_count += (Index)cond3 + (Index)cond1;

				if(! cond3)
					break;
			}
		}
		neighbor_counts[i+1] = neighbor_count;
	}
}

static void neighbor_counts_to_offsets(Index node_count, Index* neighbor_counts){
	// TODO: SIMD
	for(Index n=1; n+1<node_count+1; n++)
		neighbor_counts[n+1]+=neighbor_counts[n];
}







//	TODO(?): make sure to minimize false sharing on edge_list
// 	might require to create separate edge lists for each thread, but those would have to lie a cache line apart in memory :/
//	Windows and Linux does provide some functionality for allocating memory at specific adresses (if memory serves well)

//	Note that threads are unlikely to write to same cache line since
//	overlapping cache lines only occur for last cache line for thread t and first cache line for thread t+1
//	but by the time thread t reaches last cache line, thread t+1 will most likely have moved on from first cache line
// 	This assumes though that every thread processes more than a cache line worth of data
//	(neighbor_offsets[to]-neighbor_offsets[from] > cacheline_size/sizeof(Node)) isn't necessarily always true for fill_in_neighbors


// See comment for count_neighbors for an explanation of the seemingly complicated nature of the iteration
static void fill_in_neighbors(Index node_count, Index* neighbor_offsets, PyArrayObject* edge_list, PyArrayObject* edge_types, Index from, Index to, int* onset_div, int* duration_div){
	for(Index i=from; i<to; i++){
		Index neighbor_count = neighbor_offsets[i+1] - neighbor_offsets[i];

		Node* src = (Node*)PyArray_GETPTR2(edge_list, 0, neighbor_offsets[i]);
		Node* dst = (Node*)PyArray_GETPTR2(edge_list, 1, neighbor_offsets[i]);
		EdgeType* types = (EdgeType*)PyArray_DATA(edge_types) + neighbor_offsets[i];

		Int eq_boundary = ((Int)i)-1;

		while(eq_boundary>=0 && onset_div[(Index)eq_boundary] == onset_div[i])
			eq_boundary--;

		Index cursor=0;

		Index j=(Index)(eq_boundary+1);

		for(;j<node_count; j++){
			bool
				cond1 = (onset_div[j] == onset_div[i]),
				cond3 = (onset_div[j] == onset_div[i] + duration_div[i]);

			src[cursor] = Index_To_Node(i);
			dst[cursor] = Index_To_Node(j);
			types[cursor] = Onset_Eq;

			cursor+= (Index)cond1;

			if(cursor == neighbor_count)
				break;

			src[cursor] = Index_To_Node(i);
			dst[cursor] = Index_To_Node(j);
			types[cursor] = Onset_End_Eq;

			cursor+= (Index)cond3;

			if((!cond1) | (cursor == neighbor_count))
				break;
		}

		Index j_star = j;

		if(cursor == neighbor_count)
			continue;

		for(;j<node_count; j++){
			bool
				cond2 = (onset_div[j] < onset_div[i] + duration_div[i]);

			src[cursor] = Index_To_Node(i);
			dst[cursor] = Index_To_Node(j);
			types[cursor] = Inbetween;

			

			cursor+= (Index)cond2;

			if((!cond2) | (cursor == neighbor_count))
				break;
		}

		if(cursor == neighbor_count)
			continue;

		if((j < node_count) & (onset_div[j] > onset_div[i] + duration_div[i])){
			int closest_from_above = onset_div[j];

			for(;j < node_count; j++){
				src[cursor] = Index_To_Node(i);
				dst[cursor] = Index_To_Node(j);
				types[cursor] = Closest_End;

				bool
					cond4 = (onset_div[j] == closest_from_above);

				

				cursor+= (Index)cond4;

				if((!cond4) | (cursor == neighbor_count))
					break;
			}
		}
		else{
			if(j == j_star)
				j++;

			for(;j < node_count; j++){
				bool
					cond1 = (onset_div[j] == onset_div[i]),
					cond3 = (onset_div[j] == onset_div[i] + duration_div[i]);

				

				

				src[cursor] = Index_To_Node(i);
				dst[cursor] = Index_To_Node(j);
				types[cursor] = Onset_End_Eq;

				cursor+= (Index)cond3;

				if(cursor == neighbor_count)
					break;

				src[cursor] = Index_To_Node(i);
				dst[cursor] = Index_To_Node(j);
				types[cursor] = Onset_Eq;

				cursor+= (Index)cond1;

				if((!cond3) | (cursor == neighbor_count))
					break;
			}
		}

		assert(cursor == neighbor_count);
	}
}


static PyObject* GMSamplers_compute_edge_list(PyObject* csamplers, PyObject* args){
	PyArrayObject* onset_div;
	PyArrayObject* duration_div;

	if(!PyArg_ParseTuple(args, "OO", (PyObject**)&onset_div, (PyObject**)&duration_div)){
		puts("can't compute edge list with no proper args");
		return NULL;
	}

	Index node_count = (Index)PyArray_DIM(onset_div,0);

	assert(node_count == (Index)PyArray_DIM(duration_div, 0));



	Index* neighbor_counts = (Index*)malloc(sizeof(Index)*(node_count+1));

	assert(neighbor_counts);

	// TODO: MT
	count_neighbors(node_count, neighbor_counts, 0, node_count, (int*)PyArray_DATA(onset_div), (int*)PyArray_DATA(duration_div));

	neighbor_counts[0]=0;

	neighbor_counts_to_offsets(node_count, neighbor_counts);

	const npy_intp dims[2] = {2, neighbor_counts[node_count]};
	
	PyArrayObject* edge_list = (PyArrayObject*)PyArray_SimpleNew(2, dims, Node_Eqv_In_Numpy);

	const npy_intp dim = neighbor_counts[node_count];

	PyArrayObject* edge_types = (PyArrayObject*)PyArray_SimpleNew(1, &dim, EType_Eqv_In_Numpy);

	// TODO: MT
	fill_in_neighbors(node_count, neighbor_counts, edge_list, edge_types,0, node_count, (int*)PyArray_DATA(onset_div), (int*)PyArray_DATA(duration_div));

	free(neighbor_counts);

	return PyTuple_Pack(2, edge_list, edge_types);
}


typedef struct{
	PyObject_HEAD
	Index node_count;
	Index* pre_neighbor_offsets;
	PyArrayObject* edge_list;
} Graph;


static void Graph_dealloc(Graph* graph){
	free(graph->pre_neighbor_offsets);
	Py_DECREF(graph->edge_list);
	Py_TYPE(graph)->tp_free((PyObject*)graph);
}


static PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
	Index node_count;
	PyArrayObject* edges;

	//	ASSUMPTION: edge list should be sorted in the second argument or destination

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, &node_count)){
		puts("no new graph without correct args");
		return NULL;
	}

	// TODO: check edges.dtype in {np.uint32, np.uint64}


	Graph* graph = (Graph*) type->tp_alloc(type, 0);
	
	if(graph!=NULL){
		// NOTE: if ASSUMPTION in count_neighbors is wrong, memory needs to be zeroed out before starting counting
		graph->pre_neighbor_offsets = (Index*)malloc((node_count+1)*sizeof(Index));



		assert(graph->pre_neighbor_offsets);
	}

	return (PyObject*)graph;
}










static int Graph_init(Graph* graph, PyObject* args, PyObject* kwds){
	PyArrayObject* edges;
	Index node_count;

	//	ASSUMPTION: edge list should be sorted in the second argument

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, &node_count)){
		puts("couldn't parse edge list");
		return -1;
	}

	graph->node_count = node_count;

	graph->edge_list = edges;


	

	
	// TODO: MT or GPU
	for(Index i=0; i<graph->node_count+1; i++){
    	graph->pre_neighbor_offsets[i]=0;
    }

    Index edge_count = (Index)PyArray_DIM(edges, 1);

	for(Index e=0; e<edge_count; e++){
		Node dst_node = *((Node*)PyArray_GETPTR2(edges, 1, e));
		graph->pre_neighbor_offsets[Node_To_Index(dst_node)+1]++;
    }

    neighbor_counts_to_offsets(graph->node_count, graph->pre_neighbor_offsets);


	return 0;
}

static PyObject* Graph_print(Graph* graph, PyObject *Py_UNUSED(ignored)){
	for(Index i=0;i<graph->node_count+1;i++){
		printf("%u\t", graph->pre_neighbor_offsets[i]);
	}
	printf("\n");

	Py_RETURN_NONE;
}

static Node src_node_at(Graph* g, Index i){
	return *((Node*)PyArray_GETPTR2(g->edge_list, 0, i));
}

static PyMethodDef Graph_methods[] = {
	{"print", (PyCFunction)Graph_print, METH_NOARGS, "print the graph"},
	{NULL}
};

static PyTypeObject GraphType = {
	PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "GMSamplers.Graph",
    .tp_doc = PyDoc_STR("GMSamplers graph"),
    .tp_basicsize = sizeof(Graph),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Graph_new,
    .tp_init = (initproc) Graph_init,
    .tp_dealloc = (destructor) Graph_dealloc,
    .tp_methods = Graph_methods
};












static PyArrayObject* index_array_to_numpy(Index* indices, uint size){
	const npy_intp dims = size;
	PyArrayObject* np_arr = (PyArrayObject*)PyArray_SimpleNew(1, &dims, Index_Eqv_In_Numpy);

	Index* np_indices = (Index*)PyArray_DATA(np_arr);

	while(size--)
		*np_indices++ = *indices++;

	return np_arr;
}

// static void write_node_at(Node n, PyArrayObject* np_arr, npy_intp index){
// 	*((Node*)PyArray_GETPTR1(np_arr, index)) = n;
// }






static uint64 power(uint32 base, uint32 exponent){
	uint64 p = 1;

	while(exponent--)
		p*=base;

	return p;
}



static Index binary_search(Node n, Node* non_decr_list, Index size){
	Index l=0, r=size-1;

	while(l<=r){
		Index probe = (l+r)/2;

		if(non_decr_list[probe] < n)
			l=probe+1;
		else if(non_decr_list[probe] > n)
			r=probe-1;
		else
			return probe;
	}

	return size;
}



static bool is_subset_of(Node* lhs, Index lhs_size, Node* rhs, Index rhs_size){
	//ASSUMPTION: lhs and rhs are sorted in ascending order and there are no repeated elements

	Index cursor=binary_search(lhs[0], rhs, rhs_size);

	if(cursor == rhs_size)
		return false;

	for(Index i=1; i < lhs_size; i++){
		cursor+=(binary_search(lhs[i], rhs+cursor+1, rhs_size-cursor-1)+1);

		if(cursor == rhs_size)
			return false;
	}

	return true;
}



static PyObject* GMSamplers_sample_nodewise(PyObject* csamplers, PyObject* args){
	uint depth;
	Index samples_per_node;
	PyArrayObject* target_nodes = NULL;
	Graph* graph;

	if(!PyArg_ParseTuple(args, "OII|O", (PyObject**)&graph, &depth, (uint*)&samples_per_node, (PyObject**)&target_nodes)){
		printf("If you don't provide proper arguments, you can't have any neighbor sampling.\nHow can you have any neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	/* 	TODO
		disallow samples_per_node > graph->node_count, it just doesn't make sense
		
		emit warning if (maximum) total number of samples > graph->node_count

		maybe (but really only maybe) check if target nodes actually occur in graph
	*/

	PyObject* samples_per_layer = PyList_New(depth+1);
	PyObject* load_per_layer = PyList_New(depth);
	PyObject* edge_indices_between_layers = PyList_New(depth);


	if((samples_per_layer == NULL) | (load_per_layer == NULL) | (edge_indices_between_layers == NULL)){
		printf("can't create return pylists\n");

		Py_XDECREF(samples_per_layer);
		Py_XDECREF(load_per_layer);
		Py_XDECREF(edge_indices_between_layers);

		return NULL;
	}

	Index prev_size;

	HashSet load_set;

	PyArrayObject* prev_layer;

	if(target_nodes == NULL){
		PyArrayObject* init_layer = new_node_numpy(MACRO_MIN(samples_per_node,graph->node_count));

		if(init_layer == NULL){
			puts("couldn't create init layer");
			Py_DECREF(samples_per_layer);
			Py_DECREF(edge_indices_between_layers);
			Py_DECREF(load_per_layer);
			return NULL;
		}

		HashSet_new(&load_set, samples_per_node);
		HashSet_init(&load_set);

		Node* init_nodes = (Node*)PyArray_DATA(init_layer);

		for(uint sample=0; sample<samples_per_node; sample++){
			Node node_sample;

			for(;;){
				node_sample = rand()%graph->node_count; // TODO: need to make sure these are valid samples
				if(HashSet_add_node(&load_set, node_sample))
					break;
			}
			
			*init_nodes++ = node_sample;
		}

		PyList_SET_ITEM(samples_per_layer, depth, (PyObject*)init_layer);

		prev_size = samples_per_node;

		prev_layer = init_layer;
	}
	else{
		//TODO: what to do if target_nodes is not owned by the caller?
		//TODO: what if target nodes doesnt have shape (N,) or (N,1)?
		Py_INCREF(target_nodes);
		prev_size = PyArray_SIZE(target_nodes);

		PyList_SET_ITEM(samples_per_layer, depth, (PyObject*)target_nodes);

		
		HashSet_new(&load_set, prev_size);
		HashSet_init(&load_set);

		Node* raw_target_nodes = PyArray_DATA(target_nodes);

		for(uint n = 0 ; n < prev_size; n++)
			HashSet_add_node(&load_set, *raw_target_nodes++);


		prev_layer = target_nodes;
	}

	HashSet node_hash_set;
	HashSet_new(&node_hash_set, prev_size);

	HashSet node_tracker;
	HashSet_new(&node_tracker, samples_per_node);

	

	// We allocate this much upfront because it will be used all almost surely
	Index* edge_index_canvas = (Index*)malloc(sizeof(Index)*prev_size*power(samples_per_node, depth));

	assert(edge_index_canvas);
	
	for(uint layer=depth;layer>0; layer--){
		Index cursor=0;

		HashSet_init(&node_hash_set);

		


		Node* prev_layer_nodes = (Node*)PyArray_DATA(prev_layer);

		for(Index n=0; n<prev_size; n++){
			Node dst_node = prev_layer_nodes[n];


			Index offset = graph->pre_neighbor_offsets[Node_To_Index(dst_node)];
			Index pre_neighbor_count = graph->pre_neighbor_offsets[Node_To_Index(dst_node)+1]-graph->pre_neighbor_offsets[Node_To_Index(dst_node)];

			if(pre_neighbor_count <= samples_per_node){
				for(Index i=0; i<pre_neighbor_count; i++){
					Node pre_neighbor = src_node_at(graph, offset + i);

					HashSet_add_node(&node_hash_set, pre_neighbor);
					HashSet_add_node(&load_set, pre_neighbor);

					edge_index_canvas[cursor++] = offset + i;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (uint)(0.75*pre_neighbor_count)){
				Index* perm = (Index*)malloc(sizeof(Index)*pre_neighbor_count);

				assert(perm);

				for(Index i=0; i<pre_neighbor_count; i++)
					perm[i]=i;
				

				for(Index i=0; i<samples_per_node; i++){
					Index rand_i = i + rand()%(pre_neighbor_count-i);

					Node node_sample = src_node_at(graph, offset + perm[rand_i]);

					HashSet_add_node(&node_hash_set, node_sample);
					HashSet_add_node(&load_set, node_sample);

					edge_index_canvas[cursor++] = offset + perm[rand_i];

					perm[rand_i]=perm[i];
				}

				free(perm);
			}
			else{
				HashSet_init(&node_tracker);

				for(uint sample=0; sample<samples_per_node; sample++){
					Index edge_index;

					Node node_sample;

					uint attempts=1;

					for(;;){
						edge_index = rand()%pre_neighbor_count;
						node_sample = src_node_at(graph, offset + edge_index);
						if(HashSet_add_node(&node_tracker, node_sample))
							break;

						attempts++;
					}
					
					HashSet_add_node(&node_hash_set, node_sample);
					HashSet_add_node(&load_set, node_sample);
					

					edge_index_canvas[cursor++] = offset + edge_index;
				}
			}
		}

		
		PyArrayObject* edge_indices = index_array_to_numpy(edge_index_canvas, cursor);
		

		PyArrayObject* new_layer = HashSet_to_numpy(&node_hash_set);
		PyArrayObject* layer_load = HashSet_to_numpy(&load_set);

		HashSet tmp = load_set;

		load_set = node_hash_set;
		node_hash_set = tmp;


		prev_size = PyArray_SIZE(new_layer);
		prev_layer = new_layer;

		PyList_SET_ITEM(samples_per_layer, layer-1, (PyObject*)new_layer);
		PyList_SET_ITEM(load_per_layer, layer-1, (PyObject*)layer_load);
		PyList_SET_ITEM(edge_indices_between_layers, layer-1, (PyObject*)edge_indices);
	}

	HashSet_free(&node_tracker);
	HashSet_free(&load_set);
	HashSet_free(&node_hash_set);
	free(edge_index_canvas);

	return PyTuple_Pack(3, samples_per_layer, edge_indices_between_layers, load_per_layer);
}


static PyObject* GMSamplers_sample_layerwise(PyObject* csamplers, PyObject* args){
	uint depth;
	Index samplesize_per_layer;
	PyArrayObject* target_nodes = NULL;
	Graph* graph;

	if(!PyArg_ParseTuple(args, "OII|O", (PyObject**)&graph, &depth, (uint*)&samplesize_per_layer, (PyObject**)&target_nodes)){
		printf("If you don't provide proper arguments, you can't have any layerwise sampling.\nHow can you have any layerwise sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	/* 	TODO
		disallow samplesize_per_layer > graph->node_count, it just doesn't make sense
		
		emit warning if (maximum) total number of samples > graph->node_count

		maybe (but really only maybe) check if target nodes actually occur in graph
	*/

	PyObject* samples_per_layer = PyList_New(depth+1);
	PyObject* load_per_layer = PyList_New(depth);
	PyObject* edge_indices_between_layers = PyList_New(depth);


	if((samples_per_layer == NULL) | (load_per_layer == NULL) | (edge_indices_between_layers == NULL)){
		printf("can't create return pylists\n");

		Py_XDECREF(samples_per_layer);
		Py_XDECREF(load_per_layer);
		Py_XDECREF(edge_indices_between_layers);

		return NULL;
	}

	Index prev_size;

	HashSet load_set;

	PyArrayObject* prev_layer;

	if(target_nodes == NULL){
		PyArrayObject* init_layer = new_node_numpy(MACRO_MIN(samplesize_per_layer,graph->node_count));

		if(init_layer == NULL){
			puts("couldn't create init layer");
			Py_DECREF(samples_per_layer);
			Py_DECREF(edge_indices_between_layers);
			Py_DECREF(load_per_layer);
			return NULL;
		}

		HashSet_new(&load_set, samplesize_per_layer);
		HashSet_init(&load_set);

		Node* init_nodes = (Node*)PyArray_DATA(init_layer);

		for(uint sample=0; sample<samplesize_per_layer; sample++){
			Node node_sample;

			for(;;){
				node_sample = rand()%graph->node_count; // TODO: need to make sure these are valid samples
				if(HashSet_add_node(&load_set, node_sample))
					break;
			}
			
			*init_nodes++ = node_sample;
		}

		PyList_SET_ITEM(samples_per_layer, depth, (PyObject*)init_layer);

		prev_size = samplesize_per_layer;

		prev_layer = init_layer;
	}
	else{
		//TODO: what to do if target_nodes is not owned by the caller?
		//TODO: what if target nodes doesnt have shape (N,) or (N,1)?
		Py_INCREF(target_nodes);
		prev_size = PyArray_SIZE(target_nodes);

		PyList_SET_ITEM(samples_per_layer, depth, (PyObject*)target_nodes);

		
		HashSet_new(&load_set, prev_size);
		HashSet_init(&load_set);

		Node* raw_target_nodes = PyArray_DATA(target_nodes);

		for(uint n = 0 ; n < prev_size; n++)
			HashSet_add_node(&load_set, *raw_target_nodes++);


		prev_layer = target_nodes;
	}

	HashSet node_hash_set;
	HashSet_new(&node_hash_set, prev_size);
	assert(node_tracker.nodes);

	Index* edge_index_canvas = (Index*)malloc(sizeof(Index)*prev_size);

	assert(edge_index_canvas);

	Index* permutation_canvas = NULL;
	Index permutation_canvas_size=0;
	
	for(uint layer=depth;layer>0; layer--){
		Node* prev_layer_nodes = (Node*)PyArray_DATA(prev_layer);

		Index total_pre_neighbor_count = 0;

		for(Index n=0; n<PyArray_SIZE(prev_layer); n++){
			Node dst = prev_layer_nodes[n];
			total_pre_neighbor_count += graph->pre_neighbor_offsets[Node_To_Index(dst)+1]-graph->pre_neighbor_offsets[Node_To_Index(dst)];
		}

		if(permutation_canvas_size < total_pre_neighbor_count){
			free(permutation_canvas);
			permutation_canvas = (Index*)malloc(sizeof(Index)*total_pre_neighbor_count);
			permutation_canvas_size = total_pre_neighbor_count;
		}
		
		
		Index cursor=0;

		for(Index n=0; n<PyArray_SIZE(prev_layer); n++){
			Node dst = prev_layer_nodes[n];

			Index pre_neighbor_count = graph->pre_neighbor_offsets[Node_To_Index(dst)+1]-graph->pre_neighbor_offsets[Node_To_Index(dst)];
			Index offset = graph->pre_neighbor_offsets[Node_To_Index(dst)];

			for(Index i=0; i<pre_neighbor_count; i++)
				permutation_canvas[cursor++] = offset + i;
		}

		HashSet_init(&node_hash_set);

		cursor=0;
		while(cursor<total_pre_neighbor_count){
			Index rand_i = cursor + rand()%(total_pre_neighbor_count-cursor);

			edge_index_canvas[cursor] = permutation_canvas[rand_i];

			Node node_sample = src_node_at(graph, permutation_canvas[rand_i]);

			permutation_canvas[rand_i] = permutation_canvas[cursor];

			cursor++;

			HashSet_add_node(&node_hash_set, node_sample);
			HashSet_add_node(&load_set, node_sample);

			if(node_hash_set.size == samplesize_per_layer)
				break;		
		}


		// Index attempts = 0;

		// while(true){
		// 	Index rand_i = rand()%PyArray_SIZE(prev_layer);

		// 	Node dst_sample = prev_layer_nodes[rand_i];

		// 	Index offset = graph->pre_neighbor_offsets[Node_To_Index(dst_sample)];

		// 	Index pre_neighbor_count = graph->pre_neighbor_offsets[Node_To_Index(dst_sample)+1]-graph->pre_neighbor_offsets[Node_To_Index(dst_sample)];



		// 	Node* pre_neighbors = (Node*)PyArray_GETPTR2(graph->edge_list, 0, offset);

		// 	Index edge_index = rand()%pre_neighbor_count;

		// 	edge_index_canvas[node_hash_set.size] = offset + edge_index;

		// 	Node src_sample = pre_neighbors[edge_index];

		// 	bool fail = !HashSet_add_node(&node_hash_set, src_sample);

		// 	if(node_hash_set.size == samplesize_per_layer)
		// 		break;

		// 	attempts = ((Index)fail)*(attempts+1);

		// 	double diff = (double)(samplesize_per_layer - node_hash_set.size);

		// 	double expected_attempts = samplesize_per_layer/diff;
		// 	double variance_attempts = node_hash_set.size/(diff*diff);

		// 	if((double)attempts > expected_attempts + variance_attempts)
		// 		break;
		// }

		// if(node_hash_set.size < samplesize_per_layer){

		// }

		

		
		PyArrayObject* edge_indices = index_array_to_numpy(edge_index_canvas, cursor);
		

		PyArrayObject* new_layer = HashSet_to_numpy(&node_hash_set);
		PyArrayObject* layer_load = HashSet_to_numpy(&load_set);

		// we swap the 2 hashsets since node_hash_set already contains the nodes
		// that are needed for load_set in next iteration
		// NOTE: simply overwriting load_set doesn't work since that would introduce
		// a dangling pointer with load_set.keys 
		HashSet tmp = load_set;
		load_set = node_hash_set;
		node_hash_set = tmp;


		prev_size = PyArray_SIZE(new_layer);
		prev_layer = new_layer;

		PyList_SET_ITEM(samples_per_layer, layer-1, (PyObject*)new_layer);
		PyList_SET_ITEM(load_per_layer, layer-1, (PyObject*)layer_load);
		PyList_SET_ITEM(edge_indices_between_layers, layer-1, (PyObject*)edge_indices);
	}

	HashSet_free(&load_set);
	HashSet_free(&node_hash_set);
	free(edge_index_canvas);
	free(permutation_canvas);

	return PyTuple_Pack(3, samples_per_layer, edge_indices_between_layers, load_per_layer);
}


static PyMethodDef GMSamplersMethods[] = {
	{"sample_nodewise", GMSamplers_sample_nodewise, METH_VARARGS, "Random sampling within a graph through multiple layers where each pre-neighborhood of a node in a layer gets sampled separately."},
	{"sample_layerwise", GMSamplers_sample_layerwise, METH_VARARGS, "Random sampling within a graph through multiple layers where the pre-neighborhood of a layer gets sampled jointly."},
	{"compute_edge_list", GMSamplers_compute_edge_list, METH_VARARGS, "Compute edge list from onset_div and duration_div."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef GMSamplersmodule = {
	PyModuleDef_HEAD_INIT,
	"csamplers",
	NULL,
	-1,
	GMSamplersMethods
};

PyMODINIT_FUNC PyInit_csamplers(){
	import_array();
	if(PyType_Ready(&GraphType) < 0)
		return NULL;

	PyObject* module = PyModule_Create(&GMSamplersmodule);

	if(module==NULL)
		return NULL;

	Py_INCREF(&GraphType);
	

	if(PyModule_AddObject(module, "Graph", (PyObject*)&GraphType) < 0){
		Py_DECREF(&GraphType);
		Py_DECREF(module);
		return NULL;
	}

	return module;
}