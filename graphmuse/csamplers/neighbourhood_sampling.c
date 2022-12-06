// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
//#include <math.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef uint32_t uint;

typedef struct{
	uint index;
} Node;

static uint Node_hash(Node* n){
	return n->index;
}


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

static void NodeHashSet_init(NodeHashSet* node_hash_set, uint min_capacity){
	uint init_nodes_per_bucket = (min_capacity/NodeHashSet_bucket_count + 1);
	node_hash_set->capacity = NodeHashSet_bucket_count*init_nodes_per_bucket;

	node_hash_set->buckets[0].nodes = (Node*)malloc(sizeof(Node)*node_hash_set->capacity);
	node_hash_set->buckets[0].capacity = init_nodes_per_bucket;
	node_hash_set->buckets[0].tracked = 0;



	for(uint b=1; b<NodeHashSet_bucket_count; b++){
		node_hash_set->buckets[b].capacity = init_nodes_per_bucket;
		node_hash_set->buckets[b].tracked = 0;
		node_hash_set->buckets[b].nodes = node_hash_set->buckets[b-1].nodes + init_nodes_per_bucket;
	}
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

		int dir = (eviction_index < bucket_index)? 1 : -1;

		(nhs->buckets + eviction_index)->capacity--;

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

			eviction_index+=dir;
		}

		bucket_tracker->capacity++;
	}

	bucket_tracker->nodes[bucket_tracker->tracked++].index = n->index;

	return 1;
}


typedef struct{
	uint node_count;
	uint8_t* adjacency_matrix;
} Graph;

//Adjacency matrix is in row order form
static Graph create_graph(uint node_count, uint8_t (*matrix_func)(Node, Node)){
	Graph g;

	g.node_count = node_count;

	g.adjacency_matrix = (uint8_t*)malloc(sizeof(uint8_t)*node_count*node_count);

	for(Node i={0};i.index<node_count;i.index++)
		for(Node j={0};j.index<node_count;j.index++)
			g.adjacency_matrix[i.index*node_count + j.index] = matrix_func(i,j);

	return g;
}

static void print_graph(Graph g){
	for(uint i=0;i<g.node_count;i++){
		for(uint j=0; j<g.node_count;j++)
			printf("%u ", g.adjacency_matrix[i*g.node_count + j]);
		printf("\n");
	}
	printf("\n");
}

static uint8_t whatever(Node i, Node j){
	return (i.index==j.index) | (i.index==j.index*2) | (i.index==j.index*j.index) | (i.index==j.index/2);
}



static uint neighbors(Graph g, Node n, Node* dest){
	uint counter=0;
	for(uint i=0;i<g.node_count;i++){
		if(g.adjacency_matrix[g.node_count*n.index + i]){
			dest[counter].index = i;
			counter++;
		}
	}

	return counter;
}

// enum LayersType{
// 	RawMemory,
// 	Py_List_Of_Lists
// };

// typedef struct{
// 	LayersType type;
// 	void* internals;
// } Layers;

// static void add_layer(Layers l, Py_ssize_t layer_size){
// 	switch(l.type){
// 		case RawMemory:{
// 			break;
// 		}
// 		case Py_List_Of_Lists:{
// 			PyObject* list_of_lists = (PyObject*)l.internals;

// 			PyObject* layer = PyList_New(layer_size);

// 			int error = PyList_Append(list_of_lists, layer);

// 			//TODO: handle error

// 			break;
// 		}
// 	}
// }


// static void add_to_last_layer(Layers l, )





static uint random_neighbors(Graph g, Node n, uint sample_amount, Node* dest){
	//	TODO: the idea below only works if g.node_count is a power of a prime 

	// uint counter=0;

	// uint offset = rand(), increment = rand();
	// uint index = offset%g.node_count;
	
	// for(uint i=0;i<g.node_count;i++){
	// 	if(g.adjacency_matrix[g.node_count*n.index + index]){
	// 		dest[counter].index = index;
	// 		counter++;
	// 	}

	// 	if(counter==sample_amount)
	// 		break;

	// 	index = (index + increment)%g.node_count;
	// }

	// return counter;


	uint neighbor_count = neighbors(g, n, dest);

	// printf("Node %u has neighbors:\t", n.index);
	// for(uint i=0; i<neighbor_count; i++)
	// 	printf("%u ", dest[i].index);
	// printf("\n");

	if(neighbor_count==0)
		return 0;
	else if(neighbor_count>=sample_amount){
		for(uint i=0; i<sample_amount; i++){
			uint rand_index = rand()%(neighbor_count-i) + i;
			
			Node tmp = dest[rand_index];
			dest[rand_index] = dest[i];
			dest[i] = tmp;
		}
	}
	else{
		for(uint i=neighbor_count; i<sample_amount; i++){
			uint rand_index = rand()%neighbor_count;
			
			dest[i] = dest[rand_index];
		}
	}
	
	// printf("And random neighbors are:\t", n.index);
	// for(uint i=0; i<neighbor_count; i++)
	// 	printf("%u ", dest[i].index);
	// printf("\n");

	return sample_amount;
}


static void uniform_random_nodes(Graph g, Node* random_nodes_dest, uint amount){
	uint cursor = 0;

	while(cursor < amount){
		random_nodes_dest[cursor].index = rand()%g.node_count;
		cursor++;
	}
}


void print_from_to(Node* from, Node* to){
	while(from!=to){
		Node n = *from++;
		printf("%u ", n.index);
	}
	printf("\n----------------------------------------------\n");
}


static void sample_neighbors_raw_memory(Graph g, uint depth, uint samples_per_node, Node* samples_dest, uint target_count){
	Node* read_cursor = samples_dest;
	Node* write_cursor = samples_dest + target_count;



	if(target_count==0){
		uniform_random_nodes(g, samples_dest, samples_per_node);
		write_cursor = samples_dest + samples_per_node;
	}


	// Node* neighbor_memory = (Node*)malloc(sizeof(Node)*g.node_count);

	for(uint i=0; i<depth; i++){
		print_from_to(read_cursor, write_cursor);

		uint written=0;
		while(read_cursor != write_cursor){
			Node n = *read_cursor++;


			uint neighbor_count = (n.index<g.node_count)? random_neighbors(g, n, samples_per_node, write_cursor + written) : 0;

			

			if(neighbor_count==0){
				for(uint ii=0;ii<samples_per_node;ii++){
					write_cursor[written].index=(uint)-1;
					written++;

					// printf("%p\n", samples_dest);
				}
			}
			else
				written+=samples_per_node;

			
		}

		write_cursor+=written;
	}

	print_from_to(read_cursor, write_cursor);


	// free(neighbor_memory);
}


static void sample_neighbors_py_list(Graph g, uint depth, uint samples_per_node, Node* samples_dest, uint target_count){
	Node* read_cursor = samples_dest;
	Node* write_cursor = samples_dest + target_count;



	if(target_count==0){
		uniform_random_nodes(g, samples_dest, samples_per_node);
		write_cursor = samples_dest + samples_per_node;
	}

	Node* neighbor_memory = (Node*)malloc(sizeof(Node)*g.node_count);

	for(uint i=0; i<depth; i++){
		

		uint written=0;
		while(read_cursor != write_cursor){
			Node n = *read_cursor++;


			uint neighbor_count = (n.index<g.node_count)? random_neighbors(g, n, samples_per_node, write_cursor) : 0;

			

			if(neighbor_count==0){
				for(uint ii=0;ii<samples_per_node;ii++){
					write_cursor[written].index=(uint)-1;
					written++;

					// printf("%p\n", samples_dest);
				}
			}
			else
				written+=samples_per_node;

			
		}

		write_cursor+=written;
	}


	free(neighbor_memory);
}



static uint64_t power(uint32_t base, uint32_t exponent){
	uint64_t result = 1;

	while(exponent--)
		result*=base;

	return result;
}

static void NodeHashSet_print(NodeHashSet* node_hash_set){
	printf("\n-----------------------------------\n");
	for(uint b=0; b<NodeHashSet_bucket_count; b++){
		printf("\nBucket %u:\t tracked: %u | capacity: %u\n", b, node_hash_set->buckets[b].tracked, node_hash_set->buckets[b].capacity);
		for(uint n=0; n<node_hash_set->buckets[b].tracked; n++)
			printf("%u, ", node_hash_set->buckets[b].nodes[n].index);
	}
	printf("\n-----------------------------------\n");
}

int main(){
	NodeHashSet node_hash_set;

	uint indices_count = NodeHashSet_bucket_count*3;

	uint* indices = (uint*)malloc(sizeof(uint)*indices_count);

	for(uint i=0; i<indices_count; i++){
		indices[i]=2*NodeHashSet_bucket_count-1 - i/NodeHashSet_bucket_count;
	}

	NodeHashSet_init(&node_hash_set, indices_count);

	for(uint i=0; i<indices_count; i++){
		Node n;
		n.index = indices[i];
		if(NodeHashSet_add(&node_hash_set, &n)){
			puts("true");
		}
		else{
			puts("false");
		}
	}

	NodeHashSet_print(&node_hash_set);

	return 0;
}


// int main(){
// 	Graph g=create_graph(10,whatever);

// 	Node neighbor_memory[10];

// 	for(Node n={0}; n.index<10; n.index++){
// 		uint neighbor_count = neighbors(g, n, neighbor_memory);
// 		printf("Node %u has neighbors:\t", n.index);
// 		for(uint i=0;i<neighbor_count; i++)
// 			printf("%u ", neighbor_memory[i].index);
// 		printf("\n");
// 	}

// 	print_graph(g);

// 	uint depth = 3, samples_per_node = 2, target_count=2;

// 	uint total_sample_count = (target_count==0? samples_per_node : target_count)*((power(samples_per_node, depth + 1)-1)/(samples_per_node-1));

// 	Node samples[total_sample_count];
// 	// Node* samples = (Node*)malloc(sizeof(Node)*total_sample_count);

// 	samples[0].index=2;
// 	samples[1].index=4;

// 	sample_neighbors_raw_memory(g, depth, samples_per_node, samples, target_count);

// 	uint size = (target_count==0)? samples_per_node : target_count;

// 	Node* read_cursor = samples;

// 	//printf("%u\n", samples[0].index);

// 	// for(uint d=0; d<depth+1; d++){
// 	// 	for(uint i=0;i<size;i++){
// 	// 		Node n = *read_cursor++;
// 	// 		printf("%u ", n.index);
// 	// 	}
// 	// 	printf("\n-------------------------------------\n");
// 	// 	size*=samples_per_node;
// 	// }

// 	return 0;
// }
