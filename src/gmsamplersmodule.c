#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

#define PY_ARRAY_UNIQUE_SYMBOL sam_ARRAY_API
#include <ndarraytypes.h>
#include <ndarrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
#define MACRO_MIN(a,b) ((a)>=(b))? (b) : (a)

typedef uint32_t uint;
typedef uint32_t uint32;
typedef uint64_t uint64;

// these here are just used as indirection so that code can be compiled differently more easily
typedef uint32_t Node;
typedef uint32_t Index;
typedef int32_t Int;
#define Node_Eqv_In_Numpy NPY_UINT32
#define Index_Eqv_In_Numpy NPY_UINT32

#define Node_To_Index(n) (n)
#define Index_To_Node(i) (i)



static uint GMSamplers_memory_canvas_size = 256;
static void* GMSamplers_memory_canvas;



// this should be fine actually since it is used on random nodes
static uint Node_hash(Node n){
	return (uint)n;
}


static bool is_close(float a, float b){
	return fabs(a-b)<=1E-6;
}

// TODO: ask Manos, should pitch really be float? Not sure if equality works as expected
static bool adjacency_map(Index i, Index j, float* onset_beat, float* duration_beat, float* pitch){
	int cond1 = is_close(onset_beat[i], onset_beat[j]) & (pitch[i] != pitch[j]);

	int cond2 = is_close(onset_beat[i] + duration_beat[i], onset_beat[j]);

	int cond3 = (onset_beat[i] < onset_beat[j]) & (onset_beat[j] < onset_beat[i] + duration_beat[i]);

	return cond1 | cond2 | cond3;
}


static void count_neighbors(Index from, Index to, Index node_count, Index* neighbor_counts, float* onset_beat, float* duration_beat, float* pitch){
	for(Index i=from; i<to; i++)
		for(Index j=0; j<node_count; j++)
			// need to account that neighbor_counts eventually gets accumulated to neighbor_offsets which is 0 at index 0
			// hence i+1
			neighbor_counts[i+1]+=(Index)adjacency_map(i,j, onset_beat, duration_beat, pitch);
}

static void fill_in_neighbors(Index from, Index to, Index node_count, Index* neighbor_offsets, PyArrayObject* edge_list, float* onset_beat, float* duration_beat, float* pitch){
	// to is excluded from range
	for(Index i=from; i<to; i++){
		Node* src = (Node*)PyArray_GETPTR2(edge_list, 0, neighbor_offsets[i]);
		Node* dst = (Node*)PyArray_GETPTR2(edge_list, 1, neighbor_offsets[i]);

		Index neighbor_count = neighbor_offsets[i+1] - neighbor_offsets[i];

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


typedef struct{
	PyObject_HEAD
	Index node_count;
	Index* neighbor_offsets;
	PyArrayObject* edge_list;
} Graph;



static void Graph_dealloc(Graph* graph){
	free(graph->neighbor_offsets);
	Py_DECREF(graph->edge_list);
	Py_TYPE(graph)->tp_free((PyObject*)graph);
}

static PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
	Index node_count;
	PyArrayObject* edges;

	//	ASSUMPTION: edge list should be sorted in the first argument

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, &node_count)){
		puts("no new graph without correct args");
		return NULL;
	}

	// TODO: check edges.dtype in {np.uint32, np.uint64}


	Graph* graph = (Graph*) type->tp_alloc(type, 0);
	
	if(graph!=NULL){
		graph->neighbor_offsets = (Index*)calloc(node_count+1,sizeof(Index));

		assert(graph->neighbor_offsets);
	}

	return (PyObject*)graph;
}

static int Graph_init(Graph* graph, PyObject* args, PyObject* kwds){
	PyArrayObject* edges;
	uint node_count;

	//	ASSUMPTION: edge list should be sorted in the first argument

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, &node_count)){
		puts("couldn't parse edge list");
		return -1;
	}

	graph->node_count = node_count;

	graph->edge_list = edges;

	uint edge_count = PyArray_DIM(edges, 1);
	Node* src = (Node*)PyArray_DATA(edges);
	
	// TODO: MT or GPU
	for(uint e=0; e<edge_count; e++){
    	graph->neighbor_offsets[Node_To_Index(src[e])+1]++;
    }

    // TODO: SIMD
	for(uint n=1; n+1<graph->node_count+1; n++)
		graph->neighbor_offsets[n+1]+=graph->neighbor_offsets[n];


	return 0;
}

static PyObject* Graph_print(Graph* graph, PyObject *Py_UNUSED(ignored)){
	for(Index i=0;i<graph->node_count;i++){
		printf("%u:\t", i);
		Index offset = graph->neighbor_offsets[i];
		Index neighbor_count = graph->neighbor_offsets[i+1] - graph->neighbor_offsets[i];

		Node* neighbors = (Node*)PyArray_GETPTR2(graph->edge_list, 1, offset);

		for(Index n=0; n<neighbor_count; n++)
			printf("%u ", neighbors[n]);

		printf("\n");
	}
	printf("\n");

	Py_RETURN_NONE;
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


/* 	TODO
		figure out how to do second hashing into a NodeTracker or how to replace linear search
		issue is that size changes
			- 	scale factors coprime to size (in order to visit all indices "randomly") would have to be computed every time size changes
				(size - 1 is always coprime with size) 
			- 	not sure how close remainders are if size increases, might make hashing pointless
*/


typedef struct{
	Index capacity;
	Index tracked;
	Node* nodes;
} NodeTracker;


static Index NodeTracker_index(NodeTracker* nt, Node n){
	for(Index i=0; i<nt->tracked; i++)
		if(n == nt->nodes[i])
			return i;

	return nt->capacity;
}

static bool NodeTracker_add_succesfully(NodeTracker* nt, Node n){
	if(NodeTracker_index(nt, n) < nt->capacity)
		return false;

	nt->nodes[nt->tracked++] = n;
	return true;
}

#define NodeHashSet_bucket_count 23 //should be prime

typedef struct{
	Index capacity;
	Index size;
	NodeTracker buckets[NodeHashSet_bucket_count];
}NodeHashSet;

// ASSUMPTION: capacity is already set
static void NodeHashSet_init(NodeHashSet* node_hash_set){
	assert(node_hash_set->capacity > 0);
	assert(node_hash_set->capacity % NodeHashSet_bucket_count == 0);
	assert(node_hash_set->buckets[0].nodes);

	uint nodes_per_bucket = node_hash_set->capacity/NodeHashSet_bucket_count;
	
	node_hash_set->buckets[0].capacity = nodes_per_bucket;
	node_hash_set->buckets[0].tracked = 0;

	node_hash_set->size=0;

	for(uint b=1; b<NodeHashSet_bucket_count; b++){
		node_hash_set->buckets[b].capacity = nodes_per_bucket;
		node_hash_set->buckets[b].tracked = 0;
		node_hash_set->buckets[b].nodes = node_hash_set->buckets[b-1].nodes + nodes_per_bucket;
	}
}

static void NodeHashSet_new_noinit(NodeHashSet* node_hash_set, Index min_capacity){
	Index init_nodes_per_bucket = (min_capacity/NodeHashSet_bucket_count + 1);
	Index capacity = NodeHashSet_bucket_count*init_nodes_per_bucket;
	node_hash_set->buckets[0].nodes = (Node*)malloc(sizeof(Node)*capacity);

	node_hash_set->capacity = capacity;
}

static void NodeHashSet_new(NodeHashSet* node_hash_set, Index min_capacity){
	NodeHashSet_new_noinit(node_hash_set, min_capacity);

	NodeHashSet_init(node_hash_set);
}

static uint NodeHashSet_capacity(NodeHashSet* node_hash_set){
	uint capacity=0;

	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		capacity+=node_hash_set->buckets[b].capacity;

	node_hash_set->capacity = capacity;

	return capacity;
}

static uint NodeHashSet_size(NodeHashSet* node_hash_set){
	uint size=0;

	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		size+=node_hash_set->buckets[b].tracked;

	node_hash_set->size = size;

	return size;
}

static bool NodeHashSet_is_full(NodeHashSet* nhs){
	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		if(nhs->buckets[b].tracked < nhs->buckets[b].capacity)
			return false;

	return true;
}


static bool NodeHashSet_add_succesfully(NodeHashSet* nhs, Node n){
	int bucket_index = (int)Node_hash(n)%NodeHashSet_bucket_count;
	
	NodeTracker* bucket_tracker = nhs->buckets+bucket_index;

	// Node* memory_offset = nhs->buckets[0].nodes;

	// bucket_tracker->nodes += memory_offset;

	Index index = NodeTracker_index(bucket_tracker, n);

	// bucket_tracker->nodes -= memory_offset;

	if(index < bucket_tracker->capacity){
		return false;
	}

	if(nhs->size == nhs->capacity){
		const uint grow_factor = 2;

		Node* new_nodes = (Node*)malloc(sizeof(Node)*grow_factor*nhs->capacity);

		if(new_nodes == NULL){
			// TODO: should be handled with an error msg
			puts("oops no new memory");
			return false;
		}

		printf("resizing from %u to %u\n", nhs->capacity, nhs->capacity*grow_factor);

		Index cursor=0;

		Node* mem_to_free = nhs->buckets[0].nodes;

		for(uint b=0; b<NodeHashSet_bucket_count; b++){
			NodeTracker* bucket_tracker = nhs->buckets + b;

			bucket_tracker->capacity *= grow_factor;

			Node* backup = bucket_tracker->nodes;

			bucket_tracker->nodes = new_nodes + cursor;			

			for(Index t=0; t<bucket_tracker->tracked; t++)
				bucket_tracker->nodes[t] = backup[t];

			cursor+=bucket_tracker->capacity;
		}

		free(mem_to_free);


		nhs->capacity *= grow_factor;
	}

	if(bucket_tracker->tracked == bucket_tracker->capacity){
		int eviction_index = bucket_index;

		for(int b=bucket_index+1; b<NodeHashSet_bucket_count;b++){
			NodeTracker* bt = nhs->buckets+b;

			if(bt->tracked<bt->capacity){
				eviction_index = b;
				break;
			}
		}

		if(eviction_index == bucket_index){
			for(int b=bucket_index-1; b>=0;b--){
				NodeTracker* bt = nhs->buckets+b;

				if(bt->tracked<bt->capacity){
					eviction_index = b;
					break;
				}
			}
		}

		assert(eviction_index != bucket_index);

		(nhs->buckets+eviction_index)->capacity--;

		int dir = (eviction_index < bucket_index)? 1 : -1;

		while(eviction_index != bucket_index){
			NodeTracker* full_bt = nhs->buckets + eviction_index + dir;
			NodeTracker* bt = nhs->buckets + eviction_index;

			NodeTracker* indirect_bt;
			Int write_index, read_index;

			/*
				The idea here is:
					1. take the bucket with the higher address
					2. read in the direction from the end of the bucket
						so 	if dir = 1 = ->, read at last index
							if dir = -1 = <-, read at first index
					3. write in the opposite direction, this time specifically at the end of non-empty bucket
						from the perspective of higher address bucket, hence the -1 in first branch
			
			if(dir==1){
				indirect_bt = full_bt;
				read_index = full_bt->tracked-1;
				write_index = -1;
				
			}
			else{
				indirect_bt = bt;
				read_index = 0;
				write_index = bt->tracked;
				
			}
			*/


			indirect_bt = MACRO_MAX(full_bt, bt); //TODO: should be completely branchless otherwise the branch in the comment above can be used for "more" readability
			read_index = ((Index)(dir==1))*indirect_bt->tracked - (Index)(dir==1);
			write_index = ((Index)(dir!=1))*indirect_bt->tracked - (Index)(dir==1);

			// indirect_bt->nodes+=memory_offset;

			indirect_bt->nodes[write_index] = indirect_bt->nodes[read_index];

			indirect_bt->nodes-=dir;

			// indirect_bt->nodes-=memory_offset;

			/* 	NOTE: 	this isn't necessary since intermediate buckets keep their capacity
						only the first and last bucket tradeoff capacity
						however, this is left in a comment for completion
			
			full_bt->capacity++;
			bt->capacity--;
			*/

			eviction_index+=dir;
		}

		bucket_tracker->capacity++;
	}

	// bucket_tracker->nodes += memory_offset;

	bucket_tracker->nodes[bucket_tracker->tracked++] = n;

	// bucket_tracker->nodes -= memory_offset;

	nhs->size++;

	return true;
}


static PyObject* new_node_numpy(uint size){
	const npy_intp dims = size;
	return PyArray_SimpleNew(1, &dims, Node_Eqv_In_Numpy);
}

static PyObject* index_array_to_numpy(Index* indices, uint size){
	const npy_intp dims = size;
	PyObject* np_arr = PyArray_SimpleNew(1, &dims, Index_Eqv_In_Numpy);

	Index* np_indices = (Index*)PyArray_DATA(np_arr);

	while(size--)
		*np_indices++ = *indices++;

	return np_arr;
}

// static void write_node_at(Node n, PyArrayObject* np_arr, npy_intp index){
// 	*((Node*)PyArray_GETPTR1(np_arr, index)) = n;
// }



static PyObject* NodeHashSet_to_numpy(NodeHashSet* node_hash_set){
	PyObject* np_arr = new_node_numpy(NodeHashSet_size(node_hash_set));

	Node* copy_dst = PyArray_DATA(np_arr);

	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		//TODO: could probably be optimized with memcpy or copying with byte size different from sizeof(Node)
		for(Index n=0; n<node_hash_set->buckets[b].tracked; n++)
			*copy_dst++ = node_hash_set->buckets[b].nodes[n];
		

	return np_arr;
}


static uint64 power(uint32 base, uint32 exponent){
	uint64 p = 1;

	while(exponent--)
		p*=base;

	return p;
}



static PyObject* GMSamplers_sample_neighbors(PyObject* csamplers, PyObject* args){
	uint depth, samples_per_node;
	PyObject* target_nodes = Py_None;
	Graph* graph;

	if(!PyArg_ParseTuple(args, "OII|O", (PyObject**)&graph, &depth, &samples_per_node, (PyObject**)&target_nodes)){
		printf("If you don't provide proper arguments, you can't have any neighbor sampling.\nHow can you have any neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	/* 	TODO
		disallow samples_per_node > graph->node_count, it just doesn't make sense
		
		emit warning if (maximum) total number of samples > graph->node_count

		check if numpy.dtype is numpy.uint32 or numpy.uint64

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

	uint prev_size;

	NodeTracker node_tracker = {};
	node_tracker.capacity = samples_per_node;

	NodeHashSet node_hash_set;
	NodeHashSet load_set;


	PyObject* prev_layer;

	if(target_nodes == Py_None){
		PyObject* init_layer = new_node_numpy(MACRO_MIN(samples_per_node,graph->node_count));

		if(init_layer == NULL){
			puts("couldn't create init layer");
			Py_DECREF(samples_per_layer);
			Py_DECREF(edge_indices_between_layers);
			Py_DECREF(load_per_layer);
			return NULL;
		}

		NodeHashSet_new(&load_set, samples_per_node*(samples_per_node+1));

		Node* init_nodes = (Node*)PyArray_DATA(init_layer);

		for(uint sample=0; sample<samples_per_node; sample++){
			Node node_sample;

			for(;;){
				node_sample = rand()%graph->node_count;
				if(NodeHashSet_add_succesfully(&load_set, node_sample))
					break;
			}
			
			*init_nodes++ = node_sample;
		}

		PyList_SET_ITEM(samples_per_layer, depth, init_layer);

		prev_size = samples_per_node;

		prev_layer = init_layer;
	}
	else{
		//TODO: what to do if target_nodes is not owned by the caller?
		//TODO: what if target nodes doesnt have shape (N,) or (N,1)?
		Py_INCREF(target_nodes);
		prev_size = PyArray_DIM(target_nodes, 0);

		PyList_SET_ITEM(samples_per_layer, depth, target_nodes);

		
		NodeHashSet_new(&load_set, prev_size*(samples_per_node+1));

		Node* raw_target_nodes = PyArray_DATA(target_nodes);

		for(uint n = 0 ; n < prev_size; n++)
			NodeHashSet_add_succesfully(&load_set, *raw_target_nodes++);


		prev_layer = target_nodes;
	}

	// TODO: 	allocation is the wrong way around, I think
	//			currently, NodeSets get allocated maximum total sample count and edge_index_canvas gets resized if needed
	//			but edge_index_canvas does (almost) grow to maximum total sample count since edge indices are not hashed to maintain uniqueness within a layer
	//			(exception is if neighbor_count < sample_size for some node)
	//			however, NodeSets can remain relatively small if src nodes share a lot of neighbors
	//			HOWEVER, load_set and node_hash_set trade off memory

	NodeHashSet_new_noinit(&node_hash_set, prev_size*samples_per_node);

	node_tracker.nodes = (Node*)malloc(samples_per_node*sizeof(Node));
	node_tracker.capacity = samples_per_node;
	assert(node_tracker.nodes);

	Index* edge_index_canvas = (Index*)malloc(sizeof(Index)*prev_size*power(samples_per_node, depth));

	assert(edge_index_canvas);
	
	for(uint layer=depth;layer>0; layer--){
		// if((samples_per_node+1)*prev_size > NodeHashSet_capacity(load_set)){
		// 	Node* backup = 

		// 	free(edge_index_canvas);
		// 	edge_index_canvas_size = samples_per_node*prev_size;
		// 	edge_index_canvas = (Index*)malloc(sizeof(Index)*edge_index_canvas_size);

		// 	assert(edge_index_canvas);
		// }

		Index cursor=0;

		NodeHashSet_init(&node_hash_set);

		


		Node* prev_layer_nodes = (Node*)PyArray_DATA(prev_layer);

		for(Index n=0; n<prev_size; n++){
			Node src_node = prev_layer_nodes[n];


			Index offset = graph->neighbor_offsets[Node_To_Index(src_node)];
			Index neighbor_count = graph->neighbor_offsets[Node_To_Index(src_node)+1]-graph->neighbor_offsets[Node_To_Index(src_node)];

			Node* neighbors = (Node*)PyArray_GETPTR2(graph->edge_list, 1, offset);

			if(neighbor_count <= samples_per_node){
				for(Index i=0; i<neighbor_count; i++){
					NodeHashSet_add_succesfully(&node_hash_set, neighbors[i]);
					NodeHashSet_add_succesfully(&load_set, neighbors[i]);

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
			else if(samples_per_node > (uint)(0.75*neighbor_count)){
				Index* perm = (Index*)malloc(sizeof(Index)*neighbor_count);

				assert(perm);

				for(Index i=0; i<neighbor_count; i++)
					perm[i]=i;
				

				for(Index i=0; i<samples_per_node; i++){
					Index rand_i = i + rand()%(neighbor_count-i);

					NodeHashSet_add_succesfully(&node_hash_set, neighbors[perm[rand_i]]);
					NodeHashSet_add_succesfully(&load_set, neighbors[perm[rand_i]]);

					edge_index_canvas[cursor++] = offset + perm[rand_i];

					perm[rand_i]=i;
				}

				free(perm);
			}
			else{
				node_tracker.tracked = 0;
				
				for(uint sample=0; sample<samples_per_node; sample++){
					Index edge_index;

					Node node_sample;

					uint attempts=1;

					for(;;){
						edge_index = rand()%neighbor_count;
						node_sample = neighbors[edge_index];
						if(NodeTracker_add_succesfully(&node_tracker, node_sample))
							break;

						attempts++;
					}
					
					NodeHashSet_add_succesfully(&node_hash_set, node_sample);
					NodeHashSet_add_succesfully(&load_set, node_sample);
					

					edge_index_canvas[cursor++] = offset + edge_index;
				}
			}
		}

		/*
			Monitoring of hash set usage
		{
			double in_use_load = (double)load_set.size/(double)load_set.capacity;
			double in_use_node_hash = (double)node_hash_set.size/(double)node_hash_set.capacity;
			printf("%f %f\n", in_use_load, in_use_node_hash);

			printf("%u %u\n", load_set.capacity, node_hash_set.capacity);


			Index min_cap=-1, max_cap=0;

			for(uint b=0; b<NodeHashSet_bucket_count; b++){
				min_cap = MACRO_MIN(min_cap, load_set.buckets[b].tracked);
				max_cap = MACRO_MAX(max_cap, load_set.buckets[b].tracked);
			}

			printf("%u %u %u\n", min_cap, max_cap, max_cap-min_cap);
		}
		*/
		PyObject* edge_indices = index_array_to_numpy(edge_index_canvas, cursor);
		

		PyObject* new_layer = NodeHashSet_to_numpy(&node_hash_set);
		PyObject* layer_load = NodeHashSet_to_numpy(&load_set);

		Node* tmp_nodes = load_set.buckets[0].nodes;
		Index tmp_capacity = load_set.capacity;
		load_set = node_hash_set;
		node_hash_set.buckets[0].nodes = tmp_nodes;
		node_hash_set.capacity = tmp_capacity;

		prev_size = PyArray_DIM(new_layer, 0);
		prev_layer = new_layer;

		PyList_SET_ITEM(samples_per_layer, layer-1, new_layer);
		PyList_SET_ITEM(load_per_layer, layer-1, layer_load);
		PyList_SET_ITEM(edge_indices_between_layers, layer-1, edge_indices);
	}

	free(node_tracker.nodes);
	free(node_hash_set.buckets[0].nodes);
	free(load_set.buckets[0].nodes);
	free(edge_index_canvas);

	return PyTuple_Pack(3, samples_per_layer, edge_indices_between_layers, load_per_layer);
}


static PyMethodDef GMSamplersMethods[] = {
	{"sample_neighbors", GMSamplers_sample_neighbors, METH_VARARGS, "Random neighbor sampling within a graph through multiple layers."},
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

	

	// GMSamplers_memory_canvas = malloc(GMSamplers_memory_canvas_size);

	return module;
}