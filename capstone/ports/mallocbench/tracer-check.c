/* Positive control for mqtrace-cap.c, not a benchmark: the main thread makes 2 allocations
 * and each of 3 threads 5 (malloc, calloc, malloc, realloc, malloc); the main thread frees
 * each thread's last one after the join, so everything is freed. Every pointer passes
 * through a volatile global and every block is read back into the printed sum, so the
 * compiler cannot remove an allocation. The traced build must report MQ-DONE objects=17
 * ops=17 unfreed=0 (the native build too) and, on Capstone, runtime_allocs/runtime_frees
 * > 0 and equal: the pthread bridge's own per-thread records are set aside, not counted as
 * the program's. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *volatile sink[3][6];
static void *volatile kept[3];
static volatile unsigned long sum;

static unsigned long touch(void *volatile *slot, void *p, size_t n, int v)
{
    *slot = p;
    memset(p, v, n);
    return ((unsigned char *)*slot)[n - 1];
}

static void *work(void *arg)
{
    long i = (long)arg;
    unsigned long s = 0;
    char *a = malloc(100), *b = calloc(4, 64), *c = malloc(3000);
    s += touch(&sink[i][0], a, 100, 1) + touch(&sink[i][1], b, 256, 2) + touch(&sink[i][2], c, 3000, 3);
    c = realloc(c, 6000);
    s += touch(&sink[i][3], c, 6000, 4);
    free(sink[i][0]); free(sink[i][1]); free(sink[i][3]);
    kept[i] = malloc(48);
    s += touch(&sink[i][4], kept[i], 48, 5);
    __atomic_fetch_add(&sum, s, __ATOMIC_RELAXED);
    return NULL;
}

int main(void)
{
    pthread_t t[3];
    static void *volatile mine[2];
    unsigned long s = touch(&mine[0], malloc(32), 32, 6) + touch(&mine[1], malloc(5000), 5000, 7);
    for (long i = 0; i < 3; i++)
        if (pthread_create(&t[i], NULL, work, (void *)i)) return 2;
    for (int i = 0; i < 3; i++) pthread_join(t[i], NULL);
    for (int i = 0; i < 3; i++) free(kept[i]);
    free(mine[0]); free(mine[1]);
    printf("TRACER_CHECK_OK sum=%lu\n", s + sum);
    return 0;
}
