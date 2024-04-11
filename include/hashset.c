/*

DEPENDENCIES

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <GM_assert.h>

typedef uint64_t Key;
typedef uint32_t Value;

#define MACRO_MAX(a,b) ((a)<(b))? (b) : (a)
*/

// static Key gcd(Key a, Key b){
// 	Key rem1=a, rem2=b;

// 	while(rem2 > 0){
// 		Key tmp = rem2;
// 		rem2 = rem1%rem2;
// 		rem1 = tmp;
// 	}

// 	return rem1;
// }









typedef struct{
	Key capacity;// we only allow capacities of powers of 2

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



static void HashSet_new(HashSet* hash_set, Key expected_size){
	// if caller expects a certain size, it makes sense to create a larger initial capacity, so that operations work in O(1) and a resize will hopefully be avoided
	// using expected_size exactly will result in a resize guaranteed at least once (assuming expected_size is a good guess)
	hash_set->capacity = (Key)(expected_size/HashSet_Load_Factor_Threshold);
	hash_set->capacity = next_pow2(hash_set->capacity);

	ASSERT_POW2(hash_set->capacity);

	hash_set->keys = (Key*)malloc((sizeof(Key) + sizeof(bool))*hash_set->capacity);
	hash_set->is_set = (bool*)(hash_set->keys + hash_set->capacity);

	ASSERT(hash_set->keys);
}

static void HashSet_free(HashSet* hs){
	free(hs->keys);
}

static void HashSet_init(HashSet* hash_set){
	ASSERT(hash_set->keys);

	for(Key k = 0; k<hash_set->capacity; k++)
		hash_set->is_set[k]=false;

	hash_set->size = 0;
}



static Key HashSet_index(HashSet* hash_set, Key k){
	Key probe = k;
	const Key probe_skip = skip_hash(k, hash_set->capacity);
	for(Key attempts=0; attempts < hash_set->capacity; attempts++){
		probe = MOD_POW2(probe, hash_set->capacity);

		if(hash_set->is_set[probe]){
			if(hash_set->keys[probe] == k)
				return probe;
		}
		else
			return hash_set->capacity;

		probe+=probe_skip;
	}

	return hash_set->capacity;
}

static bool HashSet_is_in(HashSet* hash_set, Key k){
	Key probe = HashSet_index(hash_set, k);

	return probe < hash_set->capacity;
}

static void add_unsafe(HashSet* hash_set, Key k){
	Key probe = k;
	const Key probe_skip = skip_hash(k, hash_set->capacity);
	while(true){
		probe = MOD_POW2(probe, hash_set->capacity);

		if(! hash_set->is_set[probe]){
			hash_set->keys[probe] = k;
			hash_set->is_set[probe] = true;
			hash_set->size++;

			return;
		}

		probe+=probe_skip;
	}
}

static bool HashSet_add(HashSet* hash_set, Key k){
	if(hash_set->size/(double)hash_set->capacity >= HashSet_Load_Factor_Threshold){
		Key new_capacity = hash_set->capacity<<1;
		Key* new_keys = (Key*)malloc(new_capacity*(sizeof(Key)+sizeof(bool)));
		bool* new_is_set = (bool*)(new_keys+new_capacity);

		ASSERT(new_keys);

		for(Key i=0;i < new_capacity; i++)
			new_is_set[i]=false;
		

		Key* old_keys = hash_set->keys;
		Key old_capacity = hash_set->capacity;
		bool* old_is_set = hash_set->is_set;

		hash_set->keys = new_keys;
		hash_set->is_set = new_is_set;
		//hash_set->probe_skip = new_probe_skip;
		hash_set->capacity = new_capacity;
		hash_set->size=0;

		for(Key i=0; i<old_capacity; i++)
			if(old_is_set[i])
				add_unsafe(hash_set, old_keys[i]);

		free(old_keys);
	}

	Key probe = k;



	const Key probe_skip = skip_hash(k, hash_set->capacity);

	while(true){
		probe = MOD_POW2(probe, hash_set->capacity);

		if(! hash_set->is_set[probe]){
			hash_set->keys[probe] = k;
			hash_set->is_set[probe] = true;
			hash_set->size++;

			return true;
		}
		else if(hash_set->keys[probe] == k)
			return false;

		probe += probe_skip;
	}
}

static void HashSet_copy(HashSet* hs, Key* dst, size_t dst_size){
	Key counter = 0;
	for(Key i=0; i<hs->capacity; i++){
		dst[counter]=hs->keys[i];

		counter += (Key)(hs->is_set[i]);

		if(counter>=dst_size)
			return;

		if(counter == hs->size){
			#ifndef GM_DEBUG_OFF
			for(Key j=i+1; j<hs->capacity; j++)
				ASSERT(!hs->is_set[j]);
			#endif

			return;
		}
	}

	ASSERT(counter == hs->size);
}