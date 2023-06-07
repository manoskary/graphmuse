#include <stdatomic.h>
typedef atomic_flag AtomicFlag;
#define AtomicFlag_test_and_set atomic_flag_test_and_set_explicit
#define AtomicFlag_set atomic_flag_test_and_set_explicit
#define AtomicFlag_clear atomic_flag_clear_explicit

#define Atomic_load atomic_load_explicit
#define Atomic_store atomic_store_explicit

#define Atomic_compare_exchange atomic_compare_exchange_strong_explicit
#define Atomic_compare_exchange_weak atomic_compare_exchange_weak_explicit

#define Atomic_increment(Ptr, Order) atomic_fetch_add_explicit(Ptr, 1, Order)

#include <threads.h>

typedef mtx_t Mutex;
#define Mutex_init mtx_init
#define Mutex_lock mtx_lock
#define Mutex_unlock mtx_unlock

typedef cnd_t Condition;
#define Condition_init cnd_init
#define Condition_wait cnd_wait
#define Condition_is_fulfilled cnd_signal
#define Condition_is_fulfilled_yall cnd_broadcast

typedef thrd_t ThreadHandle;
#define Thread_create thrd_create
#define Thread_exit thrd_exit
#define Thread_join thrd_join







// #ifdef POSIX
// #include <pthread.h>
// typedef pthread_mutex_t Mutex;
// #define Mutex_lock pthread_mutex_lock
// #define Mutex_unlock pthread_mutex_unlock
// #define Mutex_init pthread_mutex_init
// typedef pthread_cond_t Condition;
// #define Condition_wait pthread_cond_wait
// #define Condition_is_fulfilled pthread_cond_signal
// #define Condition_is_fulfilled_yall pthread_cond_broadcast
// #define Condition_init pthread_cond_init
// typedef pthread_t ThreadHandle;
// #define Thread_exit pthread_exit
// #define Thread_create pthread_create
// #define Thread_join pthread_join
// #endif

// #ifdef WIN32
// typedef CRITICAL_SECTION Mutex;
// #define Mutex_lock EnterCriticalSection

// #endif