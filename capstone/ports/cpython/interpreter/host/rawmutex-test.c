/* Eight threads take one _PyRawMutex around a non-atomic counter. Contention
 * sends lockers through _PyRawMutex_LockSlow, which links the waiter's entry
 * into the mutex word, and the unlocker through _PyRawMutex_UnlockSlow, which
 * follows that link. The count is exact only if every exclusion held. */
#define Py_BUILD_CORE 1
#include "Python.h"
#include "pycore_lock.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

enum { THREADS = 8 };
static _PyRawMutex m;
static volatile long counter;
static long iters;

static void *work(void *arg)
{
    (void)arg;
    for (long i = 0; i < iters; i++) {
        _PyRawMutex_Lock(&m);
        long c = counter;
        for (volatile int k = 0; k < 50; k++) {
        }
        counter = c + 1;
        _PyRawMutex_Unlock(&m);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    iters = argc > 1 ? atol(argv[1]) : 100000;
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++) {
        if (pthread_create(&t[i], NULL, work, NULL)) {
            perror("pthread_create");
            return 2;
        }
    }
    for (int i = 0; i < THREADS; i++) {
        pthread_join(t[i], NULL);
    }
    printf("rawmutex: %ld of %ld\n", counter, iters * THREADS);
    return counter == iters * THREADS ? 0 : 1;
}
