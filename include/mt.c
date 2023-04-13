#ifdef POSIX
#include <pthread.h>
typedef pthread_mutex_t Mutex;
#define Mutex_lock pthread_mutex_lock
#define Mutex_unlock pthread_mutex_unlock
#define Mutex_init pthread_mutex_init
typedef pthread_cond_t Condition;
#define Condition_wait pthread_cond_wait
#define Condition_is_fulfilled pthread_cond_signal
#define Condition_is_fulfilled_yall pthread_cond_broadcast
#define Condition_init pthread_cond_init
typedef pthread_t ThreadHandle;
#define Thread_exit pthread_exit
#define Thread_create pthread_create
#define Thread_join pthread_join
#endif

#ifdef WIN32
typedef CRITICAL_SECTION Mutex;
#define Mutex_lock EnterCriticalSection

#endif