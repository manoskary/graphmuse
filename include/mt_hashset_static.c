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







// 0 is used as an empty key, so cannot be used as an actual key!
typedef struct{
	uint8_t capacity_exponent;// we only allow capacities of powers of 2
	_Atomic Key* keys;
} MT_HashSet_Static;

static Key MT_HashSet_Static_capacity(MT_HashSet_Static* hash_set){
	return (Key)(1<<hash_set->capacity_exponent);
}

static void MT_HashSet_Static_print(MT_HashSet_Static* hash_set){
	printf("{");

	for(Key i=0; i<MT_HashSet_Static_capacity(hash_set); i++){
		Key k = Atomic_load(hash_set->keys+i, memory_order_relaxed);

		if(k != 0){
			printf(" %u", k+1);
		}
	}

	printf(" }\n");
}

const double MT_HashSet_Static_Load_Threshold = 7.0/8.0; // in order to ensure O(1) operations, load factor = size/capacity < 1



static void MT_HashSet_Static_new(MT_HashSet_Static* hash_set, Key expected_size){
	// if caller expects a certain size, it makes sense to create a larger initial capacity, so that operations work in O(1) and a resize will hopefully be avoided
	// using expected_size exactly will result in a resize guaranteed at least once (assuming expected_size is a good guess)
	
	Key min_capacity = (Key)(expected_size/MT_HashSet_Static_Load_Threshold);

	hash_set->capacity_exponent = 1;

	while(MT_HashSet_Static_capacity(hash_set) < min_capacity)
		hash_set->capacity_exponent++;

	hash_set->keys = malloc(MT_HashSet_Static_capacity(hash_set)*sizeof(_Atomic Key));

	ASSERT(hash_set->keys);
}

static void MT_HashSet_Static_init(MT_HashSet_Static* hash_set){
	for(Key i=0; i<MT_HashSet_Static_capacity(hash_set); i++)
		Atomic_store(hash_set->keys+i, 0, memory_order_relaxed);
}

static void MT_HashSet_Static_free(MT_HashSet_Static* hs){
	free(hs->keys);
}



static bool MT_HashSet_Static_add(MT_HashSet_Static* hash_set, Key k){
	ASSERT(k+1!=0);
	k++;

	Key probe = k;

	for(Key attempts = 0; attempts < MT_HashSet_Static_capacity(hash_set); attempts++){
		probe = MOD_POW2(probe, MT_HashSet_Static_capacity(hash_set));

		Key expected_key = 0;

		if(Atomic_compare_exchange(hash_set->keys+probe, &expected_key, k, memory_order_relaxed, memory_order_relaxed))
			return true;

		if(expected_key == k)
			return false;

		probe++;
	}

	return false;
}

static Key MT_HashSet_Static_size(MT_HashSet_Static* hs){
	Key counter = 0;
	for(Key i=0; i<MT_HashSet_Static_capacity(hs); i++){
		Key k = Atomic_load(hs->keys+i, memory_order_relaxed);
		counter += (Key)(k != 0);
	}

	return counter;
}

static void MT_HashSet_Static_copy(MT_HashSet_Static* hs, Key* dst){
	Key counter = 0;
	for(Key i=0; i<MT_HashSet_Static_capacity(hs); i++){
		dst[counter]=Atomic_load(hs->keys+i, memory_order_relaxed);

		if(dst[counter] != 0){
			dst[counter]--;
			counter++;
		}
	}
}