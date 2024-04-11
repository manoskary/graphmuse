typedef struct{
	size_t capacity;
	uint8_t* chunks;

	size_t chunk_size;

	size_t size;
} Stack;

// static bool Stack_init(Stack* s, size_t capacity, size_t chunk_size){
// 	if(s->chunk_size*s->capacity < capacity*chunk_size){
// 		free(s->chunks);
// 	}

// 	s->chunks = malloc(chunk_size*capacity);

// 	if(s->chunks == NULL)
// 		return false;

// 	s->capacity = capacity;
// 	s->size=0;
// 	s->chunk_size = chunk_size;
// }

static void Stack_push_nocheck(Stack* s, void* chunk){
	ASSERT(s->size < s->capacity);
	memcpy((void*)(s->chunks+s->size*s->chunk_size), chunk, s->chunk_size);
	s->size++;
}

static bool Stack_push(Stack* s, void* chunk){
	if(s->size>=s->capacity){
		size_t new_cap = (size_t)(1.5*s->capacity);
		void* new_chunks = malloc(s->chunk_size*new_cap);

		if(new_chunks == NULL)
			return false;

		memcpy(new_chunks, s->chunks, s->chunk_size*s->capacity);

		s->capacity = new_cap;

		free(s->chunks);

		s->chunks = new_chunks;
	}

	Stack_push_nocheck(s, chunk);
	return true;
}

static void Stack_pop_nocheck(Stack* s, uint8_t* dest){
	ASSERT(s->size>0);

	s->size--;

	memcpy(dest, (void*)(s->chunks+s->size*s->chunk_size), s->chunk_size);
}

static bool Stack_is_empty(Stack* s){
	return s->size==0;
}

static bool Stack_pop(Stack* s, uint8_t* dest){
	if(Stack_is_empty(s))
		return false;
	
	Stack_pop_nocheck(s, dest);
	return true;
}

struct Thread_ID{
	uint8_t value;
};


typedef struct{
	Stack* job_stack;
	void* shared_data;
	void (*job_process)(void*, void*,struct Thread_ID, Stack*, Mutex*);

	Mutex mutex;
	
	Condition do_something;
	Condition stack_empty;
	Condition all_done;
	
	size_t work_in_progress;

	bool terminate;
} SynchronizationHandle;

static void job_process_nop(void* shared_args, void* local_args, struct Thread_ID ID, Stack* s, Mutex* m){
}



typedef struct{
	SynchronizationHandle* sync_handle;
	struct Thread_ID ID;
} WorkerThread;

static int worker_thread(void* arguments){
	WorkerThread* worker_thread = (WorkerThread*) arguments;

	SynchronizationHandle* sync_handle = worker_thread->sync_handle;

	Mutex_lock(&(sync_handle->mutex));

	void* job_data = NULL;
	size_t job_data_size = 0;

	while(true){
		while(!sync_handle->terminate && Stack_is_empty(sync_handle->job_stack))
			Condition_wait(&(sync_handle->do_something), &(sync_handle->mutex));

		if(sync_handle->terminate){
			Mutex_unlock(&(sync_handle->mutex));
			free(job_data);
			Thread_exit(0);
		}

		if(job_data_size < sync_handle->job_stack->chunk_size){
			job_data_size = sync_handle->job_stack->chunk_size;
			free(job_data);
			job_data = malloc(job_data_size);
		}

		if(Stack_pop(sync_handle->job_stack, job_data)){
			sync_handle->work_in_progress++;

			if(Stack_is_empty(sync_handle->job_stack))
				Condition_is_fulfilled(&(sync_handle->stack_empty));

			Mutex_unlock(&(sync_handle->mutex));

			sync_handle->job_process(sync_handle->shared_data, job_data, worker_thread->ID, sync_handle->job_stack, &(sync_handle->mutex));

			Mutex_lock(&(sync_handle->mutex));

			sync_handle->work_in_progress--;

			if(sync_handle->work_in_progress == 0)
				Condition_is_fulfilled(&(sync_handle->all_done));
		}

		
	}
}

typedef struct{
	ThreadHandle* thread_handles;
	WorkerThread* worker_threads;
	size_t worker_threads_count;

	SynchronizationHandle* sync_handle;
} Threadpool;



static void Threadpool_init(Threadpool* pool, uint worker_threads_count, SynchronizationHandle* sync_handle, Stack* job_stack){
	pool->worker_threads_count = worker_threads_count;
	pool->sync_handle = sync_handle;

	pool->thread_handles = (ThreadHandle*)malloc(worker_threads_count*(sizeof(ThreadHandle) + sizeof(WorkerThread)));
	pool->worker_threads = (WorkerThread*)(pool->thread_handles + worker_threads_count);

	
	Mutex_init(&(sync_handle->mutex), 0);
	Mutex_lock(&(sync_handle->mutex));

	Condition_init(&(sync_handle->stack_empty));
	Condition_init(&(sync_handle->do_something));
	Condition_init(&(sync_handle->all_done));
	

	sync_handle->job_process = job_process_nop;
	sync_handle->shared_data = NULL;
	sync_handle->work_in_progress = 0;
	sync_handle->terminate = false;

	sync_handle->job_stack = job_stack;
	job_stack->size=0;
	job_stack->capacity=0;
	job_stack->chunk_size=0;
	job_stack->chunks = NULL;

	

	for(uint t=0; t<worker_threads_count; t++){
		pool->worker_threads[t].sync_handle = sync_handle;
		pool->worker_threads[t].ID.value = t+1;

		Thread_create(pool->thread_handles+t, worker_thread, (void*)(pool->worker_threads+t));
	}
}

static bool Threadpool_prepare_jobstack(Threadpool* pool, size_t stack_capacity, size_t chunk_size){
	Stack* jobstack = pool->sync_handle->job_stack;

	size_t current_byte_count = jobstack->chunk_size*jobstack->capacity;

	if(current_byte_count < chunk_size*stack_capacity){
		free(jobstack->chunks);
		jobstack->chunks = malloc(chunk_size*stack_capacity);

		if(jobstack->chunks == NULL)
			return false;

		jobstack->capacity = stack_capacity;
	}
	else
		jobstack->capacity = current_byte_count/chunk_size;

	jobstack->size = 0;
	jobstack->chunk_size = chunk_size;

	return true;
}

static void Threadpool_wakeup_workers(Threadpool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex
	Condition_is_fulfilled_yall(&(pool->sync_handle->do_something));
	Mutex_unlock(&(pool->sync_handle->mutex));
}

static void Threadpool_waitfor_current_jobs_tofinish(Threadpool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex
	while(pool->sync_handle->work_in_progress > 0)
		Condition_wait(&(pool->sync_handle->all_done), &(pool->sync_handle->mutex));
}

static void Threadpool_participate_until_completion(Threadpool* pool){
	struct Thread_ID ID = {0};

	void* job_data = malloc(pool->sync_handle->job_stack->chunk_size);

	Mutex_lock(&(pool->sync_handle->mutex));

	while(true){
		if(Stack_pop(pool->sync_handle->job_stack, job_data)){
			if(!Stack_is_empty(pool->sync_handle->job_stack) & (pool->sync_handle->work_in_progress==0))
				Condition_is_fulfilled_yall(&(pool->sync_handle->do_something));

			Mutex_unlock(&(pool->sync_handle->mutex));

			pool->sync_handle->job_process(pool->sync_handle->shared_data, job_data, ID, pool->sync_handle->job_stack, &(pool->sync_handle->mutex));

			Mutex_lock(&(pool->sync_handle->mutex));
		}
		else if(pool->sync_handle->work_in_progress > 0){
			Threadpool_waitfor_current_jobs_tofinish(pool);
		}
		else{
			free(job_data);
			return;
		}
	}
}

static void Threadpool_waiton_workers(Threadpool* pool){
	Mutex_lock(&(pool->sync_handle->mutex));

	while(pool->sync_handle->job_stack->size > 0)
		Condition_wait(&(pool->sync_handle->stack_empty), &(pool->sync_handle->mutex));

	Threadpool_waitfor_current_jobs_tofinish(pool);
}

static void Threadpool_terminate(Threadpool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex

	pool->sync_handle->terminate = true;

	Condition_is_fulfilled_yall(&(pool->sync_handle->do_something));
	
	Mutex_unlock(&(pool->sync_handle->mutex));

	for(uint t=0; t<pool->worker_threads_count; t++)
		Thread_join(pool->thread_handles[t], NULL);
}