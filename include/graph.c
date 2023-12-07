typedef struct{
	PyObject_HEAD
	Index node_count;
	Index* pre_neighbor_offsets;
	PyArrayObject* edge_list;
	//PyArrayObject* edge_types;
} Graph;


static void Graph_dealloc(Graph* graph){
	free(graph->pre_neighbor_offsets);
	Py_DECREF(graph->edge_list);
	Py_TYPE(graph)->tp_free((PyObject*)graph);
}


static PyObject* Graph_new(PyTypeObject* type, PyObject* args, PyObject* kwds){
	Index node_count;
	PyArrayObject* edges;
	//PyArrayObject* edge_types;

	//	ASSUMPTION: edge list should be sorted in the second argument or destination

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, /*(PyObject**)&edge_types,*/ &node_count)){
		puts("no new graph without correct args");
		return NULL;
	}

	

	// TODO: check edges.dtype in {np.uint32, np.uint64}


	Graph* graph = (Graph*) type->tp_alloc(type, 0);
	
	if(graph!=NULL){
		// NOTE: if ASSUMPTION in count_neighbors is wrong, memory needs to be zeroed out before starting counting
		graph->pre_neighbor_offsets = (Index*)malloc((node_count+1)*sizeof(Index));



		ASSERT(graph->pre_neighbor_offsets);
	}

	return (PyObject*)graph;
}










static int Graph_init(Graph* graph, PyObject* args, PyObject* kwds){
	PyArrayObject* edges;
	//PyArrayObject* edge_types;
	Index node_count;

	//	ASSUMPTION: edge list should be sorted in the second argument

	if(!PyArg_ParseTuple(args, "OI", (PyObject**)&edges, /*(PyObject**)&edge_types,*/ &node_count)){
		puts("couldn't parse edge list");
		return -1;
	}

	graph->node_count = node_count;

	graph->edge_list = edges;
	//graph->edge_types = edge_types;

	Py_INCREF(edges);


	

	
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

static Node src_node_at(Graph* g, Index i){
	ASSERT(i < g->pre_neighbor_offsets[g->node_count]);

	return *((Node*)PyArray_GETPTR2(g->edge_list, 0, i));
}

static Node dst_node_at(Graph* g, Index i){
	ASSERT(i < g->pre_neighbor_offsets[g->node_count]);
	return *((Node*)PyArray_GETPTR2(g->edge_list, 1, i));
}

static PyObject* Graph_print(Graph* graph, PyObject *Py_UNUSED(ignored)){
	for(Index i=0;i<graph->node_count;i++){
		Index c = graph->pre_neighbor_offsets[i+1]-graph->pre_neighbor_offsets[i];
		Index o=graph->pre_neighbor_offsets[i];

		for(Index ii=0; ii<c; ii++)
			printf("(%u, %u), ", src_node_at(graph, o+ii), dst_node_at(graph, o+ii));
	}
	printf("\n");

	Py_RETURN_NONE;
}

static PyObject* Graph_preneighborhood_count(Graph* graph, PyObject* args){
	Node n;
	if(!PyArg_ParseTuple(args, "I", &n)){
		printf("If you don't provide proper arguments, you can't have any neighbor sampling.\nHow can you have any neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	Index pre_neighbor_count = graph->pre_neighbor_offsets[Node_To_Index(n)+1]-graph->pre_neighbor_offsets[Node_To_Index(n)];

	return PyLong_FromIndex(pre_neighbor_count);
}

static PyObject* Graph_edge_list(Graph* graph, void* closure){
	return (PyObject*)graph->edge_list;
}

static PyMethodDef Graph_methods[] = {
	{"print", (PyCFunction)Graph_print, METH_NOARGS, "print the graph"},
	{"preneighborhood_count", (PyCFunction)Graph_preneighborhood_count, METH_VARARGS, "get the size of the pre-neighborhood of a node within the graph"},
	{NULL}
};

static PyGetSetDef Graph_properties[] = {
	{"edge_list", Graph_edge_list, NULL, "getter for underlying edge list of Graph", NULL},
	NULL
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
    .tp_methods = Graph_methods,
    //.tp_getset = Graph_properties
};
