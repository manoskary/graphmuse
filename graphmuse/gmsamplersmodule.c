#define PY_SSIZE_T_CLEAN
#include <Python.h>


#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
#define MACRO_MIN(a,b) ((a)>=(b))? (b) : (a)

typedef uint32_t uint;




static uint GMSamplers_memory_canvas_size = 256;
static void* GMSamplers_memory_canvas;



typedef struct {
	//PyObject_HEAD
	uint index;
} Node;

static PyObject* node_to_pylong(Node n){
	return PyLong_FromUnsignedLong((unsigned long)n.index);
}

static Node pylong_to_node(PyObject* pl){
	Node n = {(uint)PyLong_AsUnsignedLong(pl)};
	return n;
}

// static int Node_init(Node* node, PyObject* args, PyObject* kwds){
// 	uint index;

// 	if(!PyArg_ParseTuple(args, "I", &index)){
// 		printf("can't init Node if args are shite\n");
// 		return -1;
// 	}

// 	node->index = index;

// 	return 0;
// }


// static PyObject* Node_index(Node* node, PyObject *Py_UNUSED(ignored)){
// 	return Py_BuildValue("I", node->index);
// }

static uint Node_hash(Node* n){
	return n->index;
}

// static PyMethodDef Node_methods[]={
// 	{"index", (PyCFunction)Node_index, METH_NOARGS, "Return the index of graph node"},
// 	{NULL}
// };


// static PyTypeObject NodeType = {
// 	PyVarObject_HEAD_INIT(NULL, 0)
//     .tp_name = "GMSamplers.Node",
//     .tp_doc = PyDoc_STR("GMSamplers graph node"),
//     .tp_basicsize = sizeof(Node),
//     .tp_itemsize = 0,
//     .tp_flags = Py_TPFLAGS_DEFAULT,
//     .tp_new = PyType_GenericNew,
//     .tp_init = (initproc) Node_init,
//     .tp_methods = Node_methods,
// };



typedef struct{
	PyObject_HEAD
	uint node_count;
	uint* row_offsets;
	Node* neighbors_per_row;
	void* memory;
	PyObject* edge_list;
} Graph;



static void Graph_dealloc(Graph* graph){
	free(graph->memory);
	Py_DECREF(graph->edge_list);
	Py_TYPE(graph)->tp_free((PyObject*)graph);
}

static PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
	uint node_count = (uint)-1;
	PyObject* edges;

	//	ASSUMPTION: edge list should be sorted in the first argument

	if(!PyArg_ParseTuple(args, "O|I", &edges, &node_count)){
		puts("no new graph without correct args");
		return NULL;
	}


	Graph* graph = (Graph*) type->tp_alloc(type, 0);
	

	

	

	if(graph!=NULL){
		if(node_count == (uint)-1){
			//	estimate node count via max on edge list
			node_count = 0;

			// Node* src;
			// Node* dst;

			uint src, dst;

			for(uint e=0; e<PyList_GET_SIZE(edges); e++){
				if(!PyArg_ParseTuple(PyList_GET_ITEM(edges, e), "II", &src, &dst)){
					printf("couldn't parse edge tuple");

					//	TODO: cleanup

					return NULL;
				}

				node_count = MACRO_MAX(node_count, src);
				node_count = MACRO_MAX(node_count, dst);
			}

			node_count++;
		}

		//	NOTE: maybe use calloc for row_counts since that memory block should 0 inited
		graph->memory = malloc(sizeof(uint)*(node_count+1) + sizeof(Node)*PyList_GET_SIZE(edges));
		graph->row_offsets = (uint*)graph->memory;
		graph->neighbors_per_row = (Node*)(graph->row_offsets + node_count+1);
		graph->node_count = node_count;

		graph->edge_list = edges;
	}

	return (PyObject*)graph;
}

static int Graph_init(Graph* graph, PyObject* args, PyObject* kwds){
	PyObject* edges;
	uint node_count;

	//	ASSUMPTION: edge list should be sorted in the first argument

	if(!PyArg_ParseTuple(args, "O|I", &edges, &node_count)){
		puts("couldn't parse edge list");
		return -1;
	}

	// ASSUMPTION: graph->node_count is already correct from Graph_new
	// same with edge_list
	//	graph->node_count = node_count;
	//	graph->edge_list = edges;

	//	NOTE: zero init for now, however, might use calloc in new
	for(uint i=0;i<graph->node_count+1; i++)
		graph->row_offsets[i]=0;

	uint cursor=0;
	for(uint e=0; e<PyList_GET_SIZE(graph->edge_list); e++){
		// Node* src;
		// Node* dst;

		uint src, dst;

		if(!PyArg_ParseTuple(PyList_GET_ITEM(graph->edge_list,e), "II", &src, &dst)){
			puts("couldn't parse edge tuple");
			return -1;
		}

		graph->row_offsets[src+1]++;
		
		// if edge is sorted in the first argument, graph init can be made much faster
		graph->neighbors_per_row[cursor].index=dst;

		cursor++;
	}

	for(uint row=1; row+1<graph->node_count+1; row++)
		graph->row_offsets[row+1]+=graph->row_offsets[row];


	// uint* cursors = (uint*)calloc(graph->node_count,sizeof(uint));

	// for(uint e=0; e<PyList_GET_SIZE(edges); e++){
	// 	Node* src;
	// 	Node* dst;

	// 	if(!PyArg_ParseTuple(PyList_GET_ITEM(edges,e), "OO", &src, &dst)){
	// 		puts("couldn't parse edge tuple");
	// 		return -1;
	// 	}

	// 	uint offset = graph->row_offsets[src->index];
		
		
	// 	// if edge is sorted in the first argument, graph init can me made much faster
	// 	graph->neighbors_per_row[offset + cursors[src->index]++]=*dst;
	// }

	// free(cursors);


	return 0;
}

static PyObject* Graph_print(Graph* graph, PyObject *Py_UNUSED(ignored)){
	for(uint i=0;i<graph->node_count;i++){
		printf("%u:\t", i);
		uint offset = graph->row_offsets[i];
		uint row_count = graph->row_offsets[i+1] - graph->row_offsets[i];

		for(uint n=0; n<row_count; n++)
			printf("%u ", graph->neighbors_per_row[offset + n].index);

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



typedef struct{
	uint capacity;
	uint tracked;
	Node* nodes;
} NodeTracker;


static uint NodeTracker_index(NodeTracker* nt, Node* n){
	for(uint i=0; i<nt->tracked; i++)
		if(n->index == nt->nodes[i].index)
			return i;

	return nt->capacity;
}

static int add_succesfully(NodeTracker* nt, Node* n){
	if(NodeTracker_index(nt, n) < nt->capacity)
		return 0;

	nt->nodes[nt->tracked++].index = n->index;
	return 1;
}

#define NodeHashSet_bucket_count 23 //should be prime

typedef struct{
	uint capacity;
	NodeTracker buckets[NodeHashSet_bucket_count];
}NodeHashSet;

static void NodeHashSet_init(NodeHashSet* node_hash_set){
	assert(node_hash_set->capacity % NodeHashSet_bucket_count == 0);

	uint nodes_per_bucket = node_hash_set->capacity/NodeHashSet_bucket_count;
	
	node_hash_set->buckets[0].capacity = nodes_per_bucket;
	node_hash_set->buckets[0].tracked = 0;

	for(uint b=1; b<NodeHashSet_bucket_count; b++){
		node_hash_set->buckets[b].capacity = nodes_per_bucket;
		node_hash_set->buckets[b].tracked = 0;
		node_hash_set->buckets[b].nodes = node_hash_set->buckets[b-1].nodes + nodes_per_bucket;
	}
}

static void NodeHashSet_new(NodeHashSet* node_hash_set, uint min_capacity){
	uint init_nodes_per_bucket = (min_capacity/NodeHashSet_bucket_count + 1);
	node_hash_set->capacity = NodeHashSet_bucket_count*init_nodes_per_bucket;
	node_hash_set->buckets[0].nodes = (Node*)malloc(sizeof(Node)*node_hash_set->capacity);

	NodeHashSet_init(node_hash_set);
}

static uint NodeHashSet_size(NodeHashSet* node_hash_set){
	uint size=0;

	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		size+=node_hash_set->buckets[b].tracked;

	return size;
}

static int NodeHashSet_add(NodeHashSet* nhs, Node* n){
	int bucket_index = (int)Node_hash(n)%NodeHashSet_bucket_count;
	
	NodeTracker* bucket_tracker = nhs->buckets+bucket_index;

	uint index = NodeTracker_index(bucket_tracker, n);

	if(index < bucket_tracker->capacity){
		return 0;
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
			int write_index, read_index;

			/*
				The idea here is:
					1. take the bucket with the higher address
					2. read in the direction from the end of the bucket
						so 	if dir = 1 = ->, read at last index
							if dir = -1 = <-, read at first index
					3. write in the oppsoite direction, this time specifically at the end of non-empty bucket
						from the perspective of higher address bucket, hence the -1 in first branch 
			*/
			if(dir==1){
				indirect_bt = full_bt;
				read_index = full_bt->capacity-1;
				write_index = -1;
				
			}
			else{
				indirect_bt = bt;
				read_index = 0;
				write_index = bt->tracked;
				
			}

			indirect_bt->nodes[write_index].index = indirect_bt->nodes[read_index].index;

			indirect_bt->nodes-=dir;

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

	bucket_tracker->nodes[bucket_tracker->tracked++].index = n->index;

	return 1;
}


static PyObject* NodeHashSet_to_pylist(NodeHashSet* node_hash_set){
	PyObject* pylist = PyList_New(NodeHashSet_size(node_hash_set));

	uint cursor=0;
	for(uint b=0; b<NodeHashSet_bucket_count; b++){
		for(uint n=0; n<node_hash_set->buckets[b].tracked; n++){
			// Node* node = PyObject_New(Node, &NodeType);
			// node->index = node_hash_set->buckets[b].nodes[n].index;

			// PyList_SET_ITEM(pylist, cursor++, (PyObject*)node);
			PyList_SET_ITEM(pylist, cursor++, node_to_pylong(node_hash_set->buckets[b].nodes[n]));
		}
	}

	return pylist;
}


static PyObject* GMSamplers_sample_neighbors_pylist(PyObject* graph_muse, PyObject* args){
	uint depth, samples_per_node;
	PyObject* target_nodes = Py_None;
	Graph* graph;

	if(!PyArg_ParseTuple(args, "OII|O", (PyObject**)&graph, &depth, &samples_per_node, &target_nodes)){
		printf("no neighbor sampling if args are shite\n");
		return NULL;
	}

	PyObject* samples_per_layer = PyList_New(depth+1);
	PyObject* load_per_layer = PyList_New(depth);


	if(samples_per_layer == NULL){
		printf("can't create return list\n");
		return NULL;
	}

	PyObject* edge_indices_between_layers = PyList_New(depth);

	if(edge_indices_between_layers == NULL){
		printf("can't create edges between layers\n");
		Py_DECREF(samples_per_layer);
		return NULL;
	}

	uint prev_size;

	NodeTracker node_tracker = {};
	node_tracker.capacity = samples_per_node;

	NodeHashSet node_hash_set;
	NodeHashSet load_set;


	if(target_nodes == Py_None){
		PyObject* init_layer = PyList_New(MACRO_MIN(samples_per_node,graph->node_count));

		if(init_layer == NULL){
			puts("couldn't create init layer");
			Py_DECREF(samples_per_layer);
			Py_DECREF(edge_indices_between_layers);
			return NULL;
		}

		node_hash_set.capacity = MACRO_MIN(samples_per_node, graph->node_count)*pow(samples_per_node, depth);


		NodeHashSet_new(&load_set, MACRO_MIN(graph->node_count, node_hash_set.capacity));

		uint upper_bound = MACRO_MIN(samples_per_node, graph->node_count);
		for(uint sample=0; sample<upper_bound; sample++){
			// Node* node_sample = (Node*) PyObject_New(Node, &NodeType);
			Node node_sample;

			for(;;){
				node_sample.index = rand()%graph->node_count;
				if(NodeHashSet_add(&load_set, &node_sample))
					break;
			}
			

			PyList_SET_ITEM(init_layer, sample, node_to_pylong(node_sample));
		}

		PyList_SET_ITEM(samples_per_layer, depth, init_layer);

		prev_size = MACRO_MIN(samples_per_node, graph->node_count);
	}
	else{
		//TODO: what to do if target_nodes is not owned by the caller? 
		Py_INCREF(target_nodes);
		prev_size = PyList_GET_SIZE(target_nodes);

		PyList_SET_ITEM(samples_per_layer, depth, target_nodes);

		
		
		node_hash_set.capacity = prev_size*pow(samples_per_node, depth);

		NodeHashSet_new(&load_set, MACRO_MIN(graph->node_count, node_hash_set.capacity));

		for(uint n=0; n<prev_size; n++){
			Node node = pylong_to_node(PyList_GET_ITEM(target_nodes, n));
			NodeHashSet_add(&load_set, &node);
			// NodeHashSet_add(&load_set, (Node*)PyList_GET_ITEM(target_nodes, n));
		}
	}

	NodeHashSet_new(&node_hash_set, MACRO_MIN(graph->node_count, node_hash_set.capacity));
	node_tracker.nodes = (Node*)malloc(samples_per_node*sizeof(Node));
	

	for(uint layer=depth;layer>0; layer--){
		PyObject* prev_layer = PyList_GET_ITEM(samples_per_layer, layer);

		//PyObject* new_layer = PyList_New(samples_per_node*prev_size);
		PyObject* edge_indices = PyList_New(samples_per_node*prev_size);
		uint cursor=0;

		NodeHashSet_init(&node_hash_set);

		for(uint n=0; n<prev_size; n++){
			// Node* node = (Node*)PyList_GET_ITEM(prev_layer, n);
			Node node = pylong_to_node(PyList_GET_ITEM(prev_layer, n));


			uint offset = graph->row_offsets[node.index];
			uint row_count = graph->row_offsets[node.index+1]-graph->row_offsets[node.index];

			
			if(row_count <= samples_per_node){
				for(uint i=0; i<row_count; i++){
					// Node* node_sample = PyObject_New(Node, &NodeType);
					uint edge_index = offset + i;
					
					Node node_sample;
					node_sample.index = graph->neighbors_per_row[edge_index].index;

					NodeHashSet_add(&node_hash_set, &node_sample);
					NodeHashSet_add(&load_set, &node_sample);

					// PyList_SET_ITEM(new_layer, cursor, (PyObject*)node_sample);

					PyList_SET_ITEM(edge_indices, cursor, PyLong_FromUnsignedLong((unsigned long)edge_index));

					cursor++;
				}
			}
			/*
				expected number of attempts to insert a unique sample into set with k elements is n/(n-k)
				for k=n*f, this results in 1/(1-f), meaning, if we want to limit the expected number of attempts
				to let's say 4, f has to be at most 3/4=0.75

				if this threshold is reached, random subset is sampled via random permutation
				this is viable since memory waste is at most 25% (for temporary storage)
			*/
			else if(samples_per_node > (uint)(0.75*row_count)){
				uint* perm = (uint*)malloc(sizeof(uint)*2*row_count);
				for(uint i=0; i<row_count; i++){
					perm[2*i]=graph->neighbors_per_row[offset+i].index;
					perm[2*i+1]=offset+i;
				}

				for(uint i=0; i<samples_per_node; i++){
					uint rand_i = i + rand()%(row_count-i);

					uint tmp_node=perm[2*i], tmp_edge_index = perm[2*i+1];

					perm[2*i]=perm[2*rand_i];
					perm[2*i+1]=perm[2*rand_i+1];

					perm[2*rand_i]=tmp_node;
					perm[2*rand_i+1]=tmp_edge_index;

					Node node_sample;
					node_sample.index = perm[2*i];
					uint edge_index = perm[2*i+1];

					NodeHashSet_add(&node_hash_set, &node_sample);
					NodeHashSet_add(&load_set, &node_sample);

					// PyList_SET_ITEM(new_layer, cursor, (PyObject*)node_sample);

					PyList_SET_ITEM(edge_indices, cursor, PyLong_FromUnsignedLong((unsigned long)edge_index));

					cursor++;
				}

				free(perm);
			}
			else{
				node_tracker.tracked = 0;
				for(uint sample=0; sample<samples_per_node; sample++){
					// Node* node_sample = PyObject_New(Node, &NodeType);
					uint edge_index;

					

					Node node_sample;

					uint attempts=1;

					for(;;){
						edge_index = offset + rand()%row_count;
						node_sample.index = graph->neighbors_per_row[edge_index].index;
						if(add_succesfully(&node_tracker, &node_sample))
							break;

						attempts++;
					}
					
					NodeHashSet_add(&node_hash_set, &node_sample);
					NodeHashSet_add(&load_set, &node_sample);
					

					// PyList_SET_ITEM(new_layer, cursor, (PyObject*)node_sample);

					PyObject* index_obj = PyLong_FromUnsignedLong((unsigned long)edge_index);

					PyList_SET_ITEM(edge_indices, cursor, index_obj);

					cursor++;
				}
			}
		}

		while(cursor < samples_per_node*prev_size){
			Py_INCREF(Py_None);
			// PyList_SET_ITEM(new_layer, cursor, Py_None);
			PyList_SET_ITEM(edge_indices, cursor, Py_None);
			cursor++;
		}

		

		PyObject* new_layer = NodeHashSet_to_pylist(&node_hash_set);
		PyObject* layer_load = NodeHashSet_to_pylist(&load_set);

		Node* tmp = load_set.buckets[0].nodes;
		load_set = node_hash_set;
		node_hash_set.buckets[0].nodes = tmp;

		prev_size = PyList_GET_SIZE(new_layer);

		PyList_SET_ITEM(samples_per_layer, layer-1, new_layer);
		PyList_SET_ITEM(load_per_layer, layer-1, layer_load);
		PyList_SET_ITEM(edge_indices_between_layers, layer-1, edge_indices);
	}

	free(node_tracker.nodes);
	free(node_hash_set.buckets[0].nodes);
	free(load_set.buckets[0].nodes);

	return PyTuple_Pack(3, samples_per_layer, edge_indices_between_layers, load_per_layer);
}



static PyMethodDef GMSamplersMethods[] = {
	{"sample_neighbors", GMSamplers_sample_neighbors_pylist, METH_VARARGS, "Random neighbor sampling within a graph through multiple layers."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef GMSamplersmodule = {
	PyModuleDef_HEAD_INIT,
	"samplers",
	NULL,
	-1,
	GMSamplersMethods
};

PyMODINIT_FUNC PyInit_samplers(){
	if(PyType_Ready(&GraphType) < 0)
		return NULL;

	// if(PyType_Ready(&NodeType) < 0)
	// 	return NULL;

	PyObject* module = PyModule_Create(&GMSamplersmodule);

	if(module==NULL)
		return NULL;

	Py_INCREF(&GraphType);
	

	if(PyModule_AddObject(module, "Graph", (PyObject*)&GraphType) < 0){
		Py_DECREF(&GraphType);
		Py_DECREF(module);
		return NULL;
	}

	// Py_INCREF(&NodeType);
	// if(PyModule_AddObject(module, "Node", (PyObject*)&NodeType) < 0){
	// 	Py_DECREF(&NodeType);
	// 	Py_DECREF(&GraphType);
	// 	Py_DECREF(module);
	// 	return NULL;
	// }

	// GMSamplers_memory_canvas = malloc(GMSamplers_memory_canvas_size);

	return module;
}


