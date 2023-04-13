// 
#include <mt.c>

typedef struct{
	size_t capacity;
	void** data;

	void* shared_data;

	size_t size;
} Queue;

static void Queue_put_nocheck(Queue* q, void* data){
	ASSERT(q->size < q->capacity);
	q->data[q->size]=data;
	q->size++;
}

static bool Queue_put(Queue* q, void* data){
	if(q->size>=q->capacity){
		size_t new_cap = (size_t)(1.5*q->capacity);
		void** new_data = (void**)malloc(sizeof(void*)*new_cap);

		if(new_data == NULL)
			return false;

		memcpy(new_data, q->data, sizeof(void*)*q->capacity);

		q->capacity = new_cap;

		free(q->data);

		q->data = new_data;
	}

	Queue_put_nocheck(q, data);
	return true;
}

static void* Queue_take_nocheck(Queue* q){
	void* ret = q->data[q->size-1];

	q->size--;

	return ret;
}

static bool Queue_is_empty(Queue* q){
	return q->size==0;
}

static void* Queue_take(Queue* q){
	if(Queue_is_empty(q))
		return NULL;
	return Queue_take_nocheck(q);
}

struct Thread_ID{
	unsigned char value;
};

struct Jobs{
	void** data;
	size_t amount;
};

const struct Jobs NO_JOBS = {NULL, 0};

typedef struct{
	Queue* job_queue;
	struct Jobs (*job_process)(void*, void*,struct Thread_ID);

	Mutex mutex;
	
	Condition do_something;
	Condition queue_empty;
	Condition all_done;
	
	size_t work_in_progress;

	bool terminate;
} SynchronizationHandle;

static struct Jobs job_process_nop(void* shared_args, void* local_args, struct Thread_ID ID){
	return NO_JOBS;
}



typedef struct{
	SynchronizationHandle* sync_handle;
	struct Thread_ID ID;
} WorkerThread;

static void* worker_thread(void* arguments){
	WorkerThread* worker_thread = (WorkerThread*) arguments;

	SynchronizationHandle* sync_handle = worker_thread->sync_handle;

	Mutex_lock(&(sync_handle->mutex));

	while(true){
		while(!sync_handle->terminate && sync_handle->job_queue->size == 0)
			Condition_wait(&(sync_handle->do_something), &(sync_handle->mutex));

		if(sync_handle->terminate){
			Mutex_unlock(&(sync_handle->mutex));
			Thread_exit(NULL);
		}

		void* job_data = Queue_take_nocheck(sync_handle->job_queue);

		sync_handle->work_in_progress++;

		if(Queue_is_empty(sync_handle->job_queue))
			Condition_is_fulfilled(&(sync_handle->queue_empty));

		Mutex_unlock(&(sync_handle->mutex));

		struct Jobs add_jobs_for_queue = sync_handle->job_process(sync_handle->job_queue->shared_data, job_data, worker_thread->ID);

		Mutex_lock(&(sync_handle->mutex));

		for(size_t j=0; j<add_jobs_for_queue.amount; j++)
			Queue_put(sync_handle->job_queue, add_jobs_for_queue.data[j]);

		sync_handle->work_in_progress--;

		if(sync_handle->work_in_progress == 0)
			Condition_is_fulfilled(&(sync_handle->all_done));
	}
}

typedef struct{
	ThreadHandle* thread_handles;
	WorkerThread* worker_threads;
	size_t worker_threads_count;

	SynchronizationHandle* sync_handle;
} ThreadPool;



static void ThreadPool_init(ThreadPool* pool, uint worker_threads_count, SynchronizationHandle* sync_handle, Queue* job_queue){
	pool->worker_threads_count = worker_threads_count;
	pool->sync_handle = sync_handle;

	pool->thread_handles = (ThreadHandle*)malloc(worker_threads_count*(sizeof(ThreadHandle) + sizeof(WorkerThread)));
	pool->worker_threads = (WorkerThread*)(pool->thread_handles + worker_threads_count);

	
	Mutex_init(&(sync_handle->mutex), NULL);
	Mutex_lock(&(sync_handle->mutex));

	Condition_init(&(sync_handle->queue_empty), NULL);
	Condition_init(&(sync_handle->do_something), NULL);
	Condition_init(&(sync_handle->all_done), NULL);
	

	sync_handle->job_process = job_process_nop;

	sync_handle->work_in_progress = 0;

	
	

	sync_handle->terminate = false;

	sync_handle->job_queue = job_queue;

	job_queue->capacity = worker_threads_count+1;
	job_queue->size=0;
	job_queue->data = (void**)malloc(sizeof(void*)*job_queue->capacity);
	job_queue->shared_data = NULL;

	for(uint t=0; t<worker_threads_count; t++){
		pool->worker_threads[t].sync_handle = sync_handle;
		pool->worker_threads[t].ID.value = t+1;

		Thread_create(pool->thread_handles+t, NULL, worker_thread, (void*)(pool->worker_threads+t));
	}
}

static void ThreadPool_wakeup_workers(ThreadPool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex
	Condition_is_fulfilled_yall(&(pool->sync_handle->do_something));
	Mutex_unlock(&(pool->sync_handle->mutex));
}

static void ThreadPool_waitfor_current_jobs_tofinish(ThreadPool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex
	while(pool->sync_handle->work_in_progress > 0)
		Condition_wait(&(pool->sync_handle->all_done), &(pool->sync_handle->mutex));
}

static void ThreadPool_participate_until_completion(ThreadPool* pool){
	struct Thread_ID ID = {0};

	while(true){
		Mutex_lock(&(pool->sync_handle->mutex));

		if(!Queue_is_empty(pool->sync_handle->job_queue)){
			void* job_data = Queue_take_nocheck(pool->sync_handle->job_queue);

			Mutex_unlock(&(pool->sync_handle->mutex));

			pool->sync_handle->job_process(pool->sync_handle->job_queue->shared_data, job_data, ID);
		}
		else{
			ThreadPool_waitfor_current_jobs_tofinish(pool);
			return;
		}
	}
}

static void ThreadPool_waiton_workers(ThreadPool* pool){
	Mutex_lock(&(pool->sync_handle->mutex));

	while(pool->sync_handle->job_queue->size > 0)
		Condition_wait(&(pool->sync_handle->queue_empty), &(pool->sync_handle->mutex));

	ThreadPool_waitfor_current_jobs_tofinish(pool);
}

static void ThreadPool_terminate(ThreadPool* pool){
	//ASSUMPTION: executing thread already owns pool->sync_handle->mutex

	pool->sync_handle->terminate = true;

	Condition_is_fulfilled_yall(&(pool->sync_handle->do_something));
	
	Mutex_unlock(&(pool->sync_handle->mutex));

	for(uint t=0; t<pool->worker_threads_count; t++)
		Thread_join(pool->thread_handles[t], NULL);
}