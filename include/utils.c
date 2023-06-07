#ifndef GM_DEBUG_OFF
#define ASSERT_POW2(N){ \
	Key ASSERT_POW2_N = N; \
	Key ASSERT_POW2_BIT_COUNTER=0; \
	while(ASSERT_POW2_N>0){ \
		ASSERT_POW2_BIT_COUNTER+=(Key)(ASSERT_POW2_N&1); \
		ASSERT_POW2_N>>=1; \
	} \
	ASSERT(ASSERT_POW2_BIT_COUNTER==1); \
}
#endif
#ifdef GM_DEBUG_OFF
#define ASSERT_POW2(N)
#endif

// ASSUMPTION: N is a power of 2
// then any odd number is co-prime with N
static Key skip_hash(Key k, Key N){
	ASSERT_POW2(N);

	return 2*k+1;
}

// dark magic from https://stackoverflow.com/questions/14291172/finding-the-smallest-power-of-2-greater-than-n
// further explanations: https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
static Key next_pow2(Key n){
	n+=(n==0);
	n--;
	n|=n>>1;
	n|=n>>2;
	n|=n>>4;
	n|=n>>8;
	n|=n>>16;
	return n+1;
}

#define MOD_POW2(K, P2) ((K)&(P2-1))

