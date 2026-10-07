#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <capstone/virtual.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/epoll.h>

extern int __clone(int (*)(void *), void *, int, void *, ...);

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static unsigned arrived;
static __thread unsigned local = 41;
static int pipefd[2];

static void *worker(void *arg)
{
    assert(local == 41);
    local = *(unsigned *)arg;
    assert(!pthread_mutex_lock(&mutex));
    ++arrived;
    assert(!pthread_cond_broadcast(&cond));
    while (arrived < 2) assert(!pthread_cond_wait(&cond, &mutex));
    assert(!pthread_mutex_unlock(&mutex));
    assert(local == *(unsigned *)arg);
    return arg;
}
static void *reader(void *arg)
{
    char byte;
    assert(read(pipefd[0], &byte, 1) == 1 && byte == 'x');
    return arg;
}
static void *writer(void *arg)
{
    usleep(10000);
    assert(write(pipefd[1], "x", 1) == 1);
    return arg;
}
static void *identity(void *arg) { assert(local == 41); return arg; }
static void *epoll_worker(void *arg)
{
    int p[2], ep = epoll_create1(EPOLL_CLOEXEC);
    assert(ep >= 0 && !pipe(p));
    struct epoll_event in = {.events = EPOLLIN};
    in.data.u64 = *(unsigned *)arg;
    assert(!epoll_ctl(ep, EPOLL_CTL_ADD, p[0], &in));
    assert(write(p[1], "x", 1) == 1);
    for (unsigned i = 0; i < 256; ++i) {
        struct epoll_event out;
        assert(epoll_wait(ep, &out, 1, 1000) == 1);
        assert(out.events & EPOLLIN);
        assert(out.data.u64 == in.data.u64);
    }
    close(ep); close(p[0]); close(p[1]);
    return arg;
}
struct shared_writer { int fd; char *bytes; };
static void *shared_write(void *opaque)
{
    struct shared_writer *w = opaque;
    for (unsigned i = 0; i < 4; ++i) assert(write(w->fd, w->bytes, 65536) == 65536);
    capstone_virtual_thread_exit(NULL);
}
static void check_file(int fd, char *buffer, char value)
{
    assert(lseek(fd, 0, SEEK_SET) == 0);
    for (unsigned i = 0; i < 4; ++i) {
        assert(read(fd, buffer, 65536) == 65536);
        for (unsigned j = 0; j < 65536; ++j) assert(buffer[j] == value);
    }
    assert(read(fd, buffer, 1) == 0);
}

int main(void)
{
    pthread_t a, b;
    unsigned values[2] = {51, 61};
    void *result;
    int tid;
    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
                CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID;
    assert(__clone(NULL, NULL, flags, NULL, &tid, NULL, &tid) == -ENOSYS);
    puts("VIRTUAL_PTHREAD_OK unsupported_clone_refused");
    assert(!pthread_create(&a, NULL, worker, &values[0]));
    assert(!pthread_create(&b, NULL, worker, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    assert(local == 41);
    puts("VIRTUAL_PTHREAD_OK tls mutex cond tagged_join");

    struct timespec deadline;
    assert(!clock_gettime(CLOCK_REALTIME, &deadline));
    deadline.tv_nsec += 1000000;
    if (deadline.tv_nsec >= 1000000000) { ++deadline.tv_sec; deadline.tv_nsec -= 1000000000; }
    assert(!pthread_mutex_lock(&mutex));
    assert(pthread_cond_timedwait(&cond, &mutex, &deadline) == ETIMEDOUT);
    assert(!pthread_mutex_unlock(&mutex));
    puts("VIRTUAL_PTHREAD_OK timed_wait");

    assert(!pipe(pipefd));
    assert(!pthread_create(&a, NULL, reader, &values[0]));
    assert(!pthread_create(&b, NULL, writer, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    close(pipefd[0]); close(pipefd[1]);
    puts("VIRTUAL_PTHREAD_OK independent_blocking_io");

    assert(!pthread_create(&a, NULL, epoll_worker, &values[0]));
    assert(!pthread_create(&b, NULL, epoll_worker, &values[1]));
    assert(!pthread_join(a, &result) && result == &values[0]);
    assert(!pthread_join(b, &result) && result == &values[1]);
    puts("VIRTUAL_PTHREAD_OK private_epoll_events");

    char *left = malloc(65536), *right = malloc(65536);
    assert(left && right);
    memset(left, 'a', 65536); memset(right, 'b', 65536);
    int fa = open("/tmp/virtual-shared-a", O_RDWR | O_CREAT | O_TRUNC, 0600);
    int fb = open("/tmp/virtual-shared-b", O_RDWR | O_CREAT | O_TRUNC, 0600);
    assert(fa >= 0 && fb >= 0);
    char *stack = mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    struct shared_writer w = {fb, right};
    long shared = capstone_virtual_thread_create(shared_write, &w, stack + 65536, tp);
    assert(shared > 0);
    for (unsigned i = 0; i < 4; ++i) assert(write(fa, left, 65536) == 65536);
    assert(!capstone_virtual_thread_join(shared));
    check_file(fa, left, 'a'); check_file(fb, right, 'b');
    close(fa); close(fb); unlink("/tmp/virtual-shared-a"); unlink("/tmp/virtual-shared-b");
    assert(!munmap(stack, 65536)); free(left); free(right);
    puts("VIRTUAL_PTHREAD_OK shared_tls_transport");

    for (unsigned i = 0; i < 64; ++i) {
        assert(!pthread_create(&a, NULL, identity, &values[0]));
        assert(!pthread_join(a, &result) && result == &values[0]);
    }
    puts("VIRTUAL_PTHREAD_OK 64_joined_lifetimes");
    return 0;
}
