/*

DEPENDENCIES

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>

typedef uint64_t Key;
typedef uint32_t Value;

#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
*/

static Key gcd(Key a, Key b){
	Key rem1=a, rem2=b;

	while(rem2 > 0){
		Key tmp = rem2;
		rem2 = rem1%rem2;
		rem1 = tmp;
	}

	return rem1;
}



typedef struct{
	Key capacity;
	Key probe_skip; //TODO: maybe compute multiple of these, so that Keys can have different probe_skips

	Key size;
	Key* keys;
	bool* is_set;
} HashSet;

static void HashSet_print(HashSet* hash_set){
	printf("{");

	Key counter=0;

	for(Key i=0; i<hash_set->capacity; i++)
		if(hash_set->is_set[i]){
			printf(" %u", hash_set->keys[i]);
			counter++;

			if(counter == hash_set->size)
				break;
		}

	printf(" }\n");
}

const double HashSet_Load_Factor_Threshold = 7.0/8.0; // in order to ensure O(1) operations, load factor = size/capacity < 1
const double HashSet_Grow_Factor = 2.0*HashSet_Load_Factor_Threshold; // this way, load factor >= 1/2 after resize

static void HashSet_new(HashSet* hash_set, Key expected_size){
	// if caller expects a certain size, it makes sense to create a larger initial capacity, so that operations work in O(1) and a resize will hopefully be avoided
	// using expected_size exactly will result in a resize guaranteed at least once (assuming expected_size is a good guess)
	hash_set->capacity = expected_size/HashSet_Load_Factor_Threshold;

	// this ensures that resizing definitely increases capacity
	hash_set->capacity = MACRO_MAX(hash_set->capacity, (Key)(1.0/(HashSet_Grow_Factor-1))+1);
	
	// we want an odd capacity in order to make totient(capacity)/capacity > 1/2 more likely
	// since a factor of 2 halves the quantity above
	hash_set->capacity += (Key)(hash_set->capacity%2==0);

	hash_set->keys = (Key*)malloc((sizeof(Key) + sizeof(bool))*hash_set->capacity);
	hash_set->is_set = (bool*)(hash_set->keys + hash_set->capacity);

	assert(hash_set->keys);

	hash_set->probe_skip = 1 + rand()%(hash_set->capacity-1);

	//TODO: Backup if totient(new_capacity)/new_capacity is too low for this loop to terminate quickly enough
	while(gcd(hash_set->probe_skip, hash_set->capacity) != 1)
		hash_set->probe_skip = 1 + rand()%(hash_set->capacity-1);
}

static void HashSet_free(HashSet* hs){
	free(hs->keys);
}

static void HashSet_init(HashSet* hash_set){
	assert(hash_set->keys);

	for(Key k = 0; k<hash_set->capacity; k++)
		hash_set->is_set[k]=false;

	hash_set->size = 0;
}



static Key HashSet_index(HashSet* hash_set, Key k){
	Key probe = k;
	for(Key attempts=0; attempts < hash_set->capacity; attempts++){
		probe = probe%hash_set->capacity;

		if(hash_set->is_set[probe]){
			if(hash_set->keys[probe] == k)
				return probe;
		}
		else
			return hash_set->capacity;

		attempts++;
		probe+=hash_set->probe_skip;
	}

	return hash_set->capacity;
}

static bool HashSet_is_in(HashSet* hash_set, Key k){
	Key probe = HashSet_index(hash_set, k);

	return probe < hash_set->capacity;
}

static void add_unsafe(HashSet* hash_set, Key k){
	Key probe = k;
	while(true){
		probe = probe%hash_set->capacity;

		if(! hash_set->is_set[probe]){
			hash_set->keys[probe] = k;
			hash_set->is_set[probe] = true;
			hash_set->size++;

			return;
		}

		probe+=hash_set->probe_skip;
	}
}

static bool HashSet_add(HashSet* hash_set, Key k){
	if(hash_set->size/(double)hash_set->capacity >= HashSet_Load_Factor_Threshold){
		Key new_capacity = (Key)(hash_set->capacity*HashSet_Grow_Factor);
		new_capacity += (Key)(new_capacity%2==0);

		Key* new_keys = (Key*)malloc(new_capacity*(sizeof(Key)+sizeof(bool)));
		bool* new_is_set = (bool*)(new_keys+new_capacity);

		assert(new_keys);

		for(Key i=0;i < new_capacity; i++)
			new_is_set[i]=false;
		

		Key new_probe_skip = 1 + rand()%(new_capacity-1);

		//TODO: Backup if totient(new_capacity)/new_capacity is too low for this loop to terminate quickly enough
		while(gcd(new_probe_skip, new_capacity) != 1)
			new_probe_skip = 1 + rand()%(new_capacity-1);

		Key* old_keys = hash_set->keys;
		Key old_capacity = hash_set->capacity;
		bool* old_is_set = hash_set->is_set;

		hash_set->keys = new_keys;
		hash_set->is_set = new_is_set;
		hash_set->probe_skip = new_probe_skip;
		hash_set->capacity = new_capacity;
		hash_set->size=0;

		for(Key i=0; i<old_capacity; i++)
			if(old_is_set[i])
				add_unsafe(hash_set, old_keys[i]);

		free(old_keys);
	}

	Key probe = k;

	while(true){
		probe = probe%hash_set->capacity;

		if(! hash_set->is_set[probe]){
			hash_set->keys[probe] = k;
			hash_set->is_set[probe] = true;
			hash_set->size++;

			return true;
		}
		else if(hash_set->keys[probe] == k)
			return false;

		probe += hash_set->probe_skip;
	}
}

static void HashSet_copy(HashSet* hs, Key* dst){
	Key counter = 0;
	for(Key i=0; i<hs->capacity; i++){
		dst[counter]=hs->keys[i];

		counter += (Key)(hs->is_set[i]);

		if(counter == hs->size)
			return;
	}

	assert(counter == hs->size);
}


/*
int main(){
	Key N = 100;

	HashSet hs;

	HashSet_new(&hs, 10);

	printf("\nsize: %lu\t cap: %lu\n", hs.size, hs.capacity);

	Key k = 1;

	for(Key i=0; i<N; i++){
		// Key k = (Key) rand()%100;
		HashSet_add(&hs, k);
		//printf(" %lu", k);
		puts("-------------------------------------------------");
		HashSet_print(&hs);
		printf("\nsize: %lu\t cap: %lu\n", hs.size, hs.capacity);

		k = (2*k+1)%50;
	}

	

	

	return 0;
}
*/