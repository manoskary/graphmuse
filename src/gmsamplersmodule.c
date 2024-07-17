

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL sam_ARRAY_API
#include <ndarraytypes.h>
// #include <ndarrayobject.h>
#include <numpy/arrayobject.h>


#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
#define MACRO_MIN(a,b) ((a)>=(b))? (b) : (a)


// these here are just used as indirection so that code can be compiled differently more easily
typedef int32_t Int;
typedef uint32_t Node;
typedef Node Index;

#define PyLong_FromIndex PyLong_FromUnsignedLong

typedef Node Key;

#define Node_Eqv_In_Numpy NPY_INT32
#define Index_Eqv_In_Numpy Node_Eqv_In_Numpy
#define Key_Eqv_In_Numpy Node_Eqv_In_Numpy
#define EType_Eqv_In_Numpy Node_Eqv_In_Numpy

typedef Node EdgeType;
#define Onset (0)
#define Consecutive (1)
#define During (2)
#define Rest (3)

#define Node_To_Index(n) ((Index)(n))
#define Index_To_Node(i) ((Node)(i))



#include <utils.c>
#include <GM_assert.h>
#ifdef Thread_Count_Arg
#include <mt.c>
#include <threadpool.c>
#endif






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
		//neighbor_counts[i+1] = neighbor_count;
		neighbor_counts[i + 1 - from] = neighbor_count;
	}
}

#ifdef Thread_Count_Arg
struct CountNeighborsArgs{
	Index node_count; 
	Index* neighbor_counts;
	Index thread_count;
	int* onset_div;
	int* duration_div;
};

#define CLS 64


static void count_neighbors_job(void* shared_args, void* local_args, struct Thread_ID thread_ID, Stack* s, Mutex* m){
	struct CountNeighborsArgs* cna = (struct CountNeighborsArgs*)shared_args;

	Index job_ID = *((Index*)local_args);

	Index from = job_ID*(cna->node_count/cna->thread_count);
	Index to = (job_ID == cna->thread_count - 1)? cna->node_count: from + cna->node_count/cna->thread_count;

	count_neighbors(cna->node_count, cna->neighbor_counts + from + job_ID*(CLS/sizeof(Index)), from, to, cna->onset_div, cna->duration_div);
}

#endif

static void neighbor_counts_to_offsets(Index node_count, Index* neighbor_counts){
	// TODO: SIMD
	for(Index n=0; n+1<node_count+1; n++)
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


static void write_node_numpy2(PyArrayObject* np_array, Index i, Index j, Node n){
	ASSERT(i < PyArray_DIM(np_array, 0) && j < PyArray_DIM(np_array, 1));

	Node* n_mem = (Node*)PyArray_GETPTR2(np_array, i, j);
	*n_mem = n;
}


// See comment for count_neighbors for an explanation of the seemingly complicated nature of the iteration
static void fill_in_neighbors(Index node_count, Index* neighbor_offsets, PyArrayObject* edge_list, PyArrayObject* edge_types, Index from, Index to, int* onset_div, int* duration_div){
	for(Index i=from; i<to; i++){
		Index neighbor_count = neighbor_offsets[i+1] - neighbor_offsets[i];

		// Node* src = (Node*)PyArray_GETPTR2(edge_list, 0, neighbor_offsets[i]);
		// Node* dst = (Node*)PyArray_GETPTR2(edge_list, 1, neighbor_offsets[i]);
		EdgeType* types = ((EdgeType*)PyArray_DATA(edge_types)) + neighbor_offsets[i];

		Int eq_boundary = ((Int)i)-1;

		while(eq_boundary>=0 && onset_div[(Index)eq_boundary] == onset_div[i])
			eq_boundary--;

		Index cursor=0;

		Index j=(Index)(eq_boundary+1);

		for(;j<node_count; j++){
			bool
				cond1 = (onset_div[j] == onset_div[i]),
				cond3 = (onset_div[j] == onset_div[i] + duration_div[i]);

			write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
			write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

			// src[cursor] = Index_To_Node(i);
			// dst[cursor] = Index_To_Node(j);

			ASSERT(cursor < neighbor_offsets[node_count]);

			types[cursor] = Onset;

			cursor+= (Index)cond1;

			if(cursor == neighbor_count)
				break;

			write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
			write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

			// src[cursor] = Index_To_Node(i);
			// dst[cursor] = Index_To_Node(j);

			ASSERT(cursor < neighbor_offsets[node_count]);
			types[cursor] = Consecutive;

			cursor+= (Index)cond3;

			if((!cond1) | (cursor == neighbor_count))
				break;
		}

		if(cursor == neighbor_count)
			continue;

		Index j_star = j;

		

		for(;j<node_count; j++){
			bool
				cond2 = (onset_div[j] < onset_div[i] + duration_div[i]);

			write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
			write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

			// src[cursor] = Index_To_Node(i);
			// dst[cursor] = Index_To_Node(j);
			ASSERT(cursor < neighbor_offsets[node_count]);
			types[cursor] = During;

			

			cursor+= (Index)cond2;

			if((!cond2) | (cursor == neighbor_count))
				break;
		}

		if(cursor == neighbor_count)
			continue;

		if((j < node_count) & (onset_div[j] > onset_div[i] + duration_div[i])){
			int closest_from_above = onset_div[j];

			for(;j < node_count; j++){
				write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
				write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

				// src[cursor] = Index_To_Node(i);
				// dst[cursor] = Index_To_Node(j);
				ASSERT(cursor < neighbor_offsets[node_count]);
				types[cursor] = Rest;

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

				

				

				write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
				write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

				// src[cursor] = Index_To_Node(i);
				// dst[cursor] = Index_To_Node(j);
				ASSERT(cursor < neighbor_offsets[node_count]);
				types[cursor] = Consecutive;

				cursor+= (Index)cond3;

				if(cursor == neighbor_count)
					break;

				write_node_numpy2(edge_list, 0, neighbor_offsets[i]+cursor, Index_To_Node(i));
				write_node_numpy2(edge_list, 1, neighbor_offsets[i]+cursor, Index_To_Node(j));

				// src[cursor] = Index_To_Node(i);
				// dst[cursor] = Index_To_Node(j);
				ASSERT(cursor < neighbor_offsets[node_count]);
				types[cursor] = Onset;

				cursor+= (Index)cond1;

				if((!cond3) | (cursor == neighbor_count))
					break;
			}
		}

		ASSERT(cursor == neighbor_count);
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

	ASSERT(node_count == (Index)PyArray_DIM(duration_div, 0));

	#ifdef Thread_Count_Arg

	const Index thread_count = Thread_Count_Arg;//(Index)omp_get_max_threads();
	Index* neighbor_counts = (Index*)malloc(sizeof(Index)*(node_count+1) + CLS*(thread_count-1));

	#else

	const Index thread_count = 1;
	Index* neighbor_counts = (Index*)malloc(sizeof(Index)*(node_count+1));

	#endif

	

	ASSERT(neighbor_counts);

	

	

	// TODO: see if you can merge the 2 parallel directives so that the main thread executes neighbor_counts_to_offsets while the worker threads wait

	#ifndef Thread_Count_Arg

	neighbor_counts[0]=0;
	count_neighbors(node_count, neighbor_counts, 0, node_count, (int*)PyArray_DATA(onset_div), (int*)PyArray_DATA(duration_div));

	neighbor_counts_to_offsets(node_count, neighbor_counts);

	#else

	GMSamplers_thread_pool->sync_handle->job_process = count_neighbors_job;

	struct CountNeighborsArgs shared_args;

	shared_args.node_count = node_count;
	shared_args.thread_count = thread_count;
	shared_args.neighbor_counts = neighbor_counts;
	shared_args.onset_div = (int*)PyArray_DATA(onset_div);
	shared_args.duration_div = (int*)PyArray_DATA(duration_div);



	//Index* local_args = (Index*)malloc(sizeof(Index)*thread_count);
	bool success = Threadpool_prepare_jobstack(GMSamplers_thread_pool, thread_count, sizeof(Index));
	ASSERT(success);

	for(Index t=0; t<thread_count; t++)
		Stack_push_nocheck(GMSamplers_thread_pool->sync_handle->job_stack, &t);
	
	GMSamplers_thread_pool->sync_handle->shared_data = (void*)(&shared_args);

	Threadpool_wakeup_workers(GMSamplers_thread_pool);

	Threadpool_participate_until_completion(GMSamplers_thread_pool);

	neighbor_counts[0]=0;

	for(Index n=1; n<node_count/thread_count+1; n++)
		neighbor_counts[n]+=neighbor_counts[n-1];

	for(Index t=1; t<thread_count-1; t++){
		Index write_from = t*(node_count/thread_count) + 1;
		Index write_to = write_from + node_count/thread_count;

		Index read_from = t*(node_count/thread_count + CLS/sizeof(Index)) + 1;

		neighbor_counts[write_from]=neighbor_counts[write_from-1]+neighbor_counts[read_from];

		for(Index n = write_from; n<write_to; n++)
			neighbor_counts[n]=neighbor_counts[n-1] + neighbor_counts[n - write_from + read_from];
	}

	Index t = thread_count-1;
	Index write_from = t*(node_count/thread_count) + 1;
	Index write_to = node_count+1;

	Index read_from = t*(node_count/thread_count + CLS/sizeof(Index)) + 1;

	neighbor_counts[write_from]=neighbor_counts[write_from-1]+neighbor_counts[read_from];

	for(Index n = write_from; n<write_to; n++)
		neighbor_counts[n]=neighbor_counts[n-1] + neighbor_counts[n - write_from + read_from];

	#endif

	const npy_intp dims[2] = {2, neighbor_counts[node_count]};
	
	PyArrayObject* edge_list = (PyArrayObject*)PyArray_SimpleNew(2, dims, Node_Eqv_In_Numpy);

	const npy_intp dim = neighbor_counts[node_count];

	PyArrayObject* edge_types = (PyArrayObject*)PyArray_SimpleNew(1, &dim, EType_Eqv_In_Numpy);



	fill_in_neighbors(node_count, neighbor_counts, edge_list, edge_types,0, node_count, (int*)PyArray_DATA(onset_div), (int*)PyArray_DATA(duration_div));

	free(neighbor_counts);

	return PyTuple_Pack(2, edge_list, edge_types);
}


static PyObject* GM_set_seed(PyObject* csamplers, PyObject* args){
	int64_t seed;

	if(!PyArg_ParseTuple(args, "I", &seed)){
		printf("If you don't provide proper arguments, you can't have any neighbor sampling.\nHow can you have any neighbor sampling if you don't provide proper arguments?\n");
		return NULL;
	}

	srand(seed);

	Py_RETURN_NONE;
}


static PyMethodDef GMSamplersMethods[] = {
	{"compute_edge_list", GMSamplers_compute_edge_list, METH_VARARGS, "Compute edge list from onset_div and duration_div."},
	{"c_set_seed", GM_set_seed, METH_VARARGS, ""},
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

	PyObject* module = PyModule_Create(&GMSamplersmodule);

	if(module==NULL)
		return NULL;


	#ifdef Thread_Count_Arg
	GMSamplers_thread_pool = (Threadpool*)malloc(sizeof(Threadpool) + sizeof(Stack) + sizeof(SynchronizationHandle));

	Stack* q = (Stack*)(GMSamplers_thread_pool+1);

	SynchronizationHandle* i = (SynchronizationHandle*)(q+1);

	Threadpool_init(GMSamplers_thread_pool, Thread_Count_Arg-1, i, q);
	#endif

	return module;
}
