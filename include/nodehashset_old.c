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

static void NodeHashSet_new(NodeHashSet* node_hash_set, Index min_capacity){
	Index init_nodes_per_bucket = (min_capacity/NodeHashSet_bucket_count + 1);
	Index capacity = NodeHashSet_bucket_count*init_nodes_per_bucket;
	node_hash_set->buckets[0].nodes = (Node*)malloc(sizeof(Node)*capacity);

	assert(node_hash_set->buckets[0].nodes);

	node_hash_set->capacity = capacity;
}

static void NodeHashSet_new_init(NodeHashSet* node_hash_set, Index min_capacity){
	NodeHashSet_new(node_hash_set, min_capacity);

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

static PyArrayObject* NodeHashSet_to_numpy(NodeHashSet* node_hash_set){
	PyArrayObject* np_arr = new_node_numpy(NodeHashSet_size(node_hash_set));

	Node* copy_dst = PyArray_DATA(np_arr);

	for(uint b=0; b<NodeHashSet_bucket_count; b++)
		//TODO: could probably be optimized with memcpy or copying with byte size different from sizeof(Node)
		for(Index n=0; n<node_hash_set->buckets[b].tracked; n++)
			*copy_dst++ = node_hash_set->buckets[b].nodes[n];
		

	return np_arr;
}