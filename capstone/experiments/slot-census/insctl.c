/* slotins's control: a loop of exactly 2 instructions per iteration (dec, jnz), run as
 *   insctl N           one thread, N iterations
 *   insctl N threads   four threads, N iterations each: two are joined, two are still parked when the process exits
 *   insctl N fork      a parked thread runs N iterations, then the process forks: the child runs 2N iterations, the
 *                      parent N more and exits with the thread still parked
 * Differences between runs with different N must match 2 x iterations exactly; the fixed start-up and exit cost
 * cancels. The parked threads check that per-thread counts reach the process line whether a thread exits before
 * the process or with it, and that a fork child counts none of its parent's other threads. */
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static long n;
static sem_t done;

static void spin(long k)
{
    if (k > 0)
        __asm__ volatile("1: dec %0\n\tjnz 1b" : "+r"(k));
}

static void *worker(void *arg)
{
    spin(n);
    return NULL;
}

static void *parked(void *arg)
{
    spin(n);
    sem_post(&done);
    for (;;) pause();
}

int main(int argc, char **argv)
{
    n = argc > 1 ? atol(argv[1]) : 0;
    const char *mode = argc > 2 ? argv[2] : "";
    pthread_t t[4];
    sem_init(&done, 0, 0);
    if (strcmp(mode, "threads") == 0) {
        for (int i = 0; i < 4; ++i) pthread_create(&t[i], NULL, i < 2 ? worker : parked, NULL);
        for (int i = 0; i < 2; ++i) pthread_join(t[i], NULL);
        for (int i = 0; i < 2; ++i) sem_wait(&done);
    } else if (strcmp(mode, "fork") == 0) {
        pthread_create(&t[0], NULL, parked, NULL);
        sem_wait(&done);
        pid_t pid = fork();
        if (pid == 0) {
            spin(2 * n);
            exit(0);
        }
        spin(n);
        waitpid(pid, NULL, 0);
    } else {
        spin(n);
    }
    return 0;
}
