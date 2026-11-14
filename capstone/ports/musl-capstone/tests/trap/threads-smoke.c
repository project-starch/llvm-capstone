#include <pthread.h>
#include <sched.h>
#ifdef THREADS_DISABLED_CONTROL
#include "libc.h"
#endif

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int objects[2];
static int visits;
static _Thread_local int private_count;

static void *worker(void *p)
{
    int *object = p;
    for (int i = 0; i < 1000; ++i) {
        *object = ++private_count;
        pthread_mutex_lock(&lock);
        ++visits;
        pthread_mutex_unlock(&lock);
        sched_yield();
    }
    return p;
}

int main(void)
{
#ifdef THREADS_DISABLED_CONTROL
    /* Reproduce the startup regression before the first pthread_create. */
    libc.can_do_threads = 0;
#endif
    pthread_t thread[2];
    void *result;
    for (int i = 0; i < 2; ++i)
        if (pthread_create(&thread[i], 0, worker, &objects[i])) return 1;
    for (int i = 0; i < 2; ++i) {
        if (pthread_join(thread[i], &result)) return 2;
        if (result != &objects[i] || objects[i] != 1000) return 3;
    }
    return visits != 2000;
}
