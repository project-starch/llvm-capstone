#include <pthread.h>
#include <sched.h>

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
