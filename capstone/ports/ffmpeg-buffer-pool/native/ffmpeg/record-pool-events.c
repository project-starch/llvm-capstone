#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "trace.h"

static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mutex;
static FILE *file;
static struct ff2_header header = {.magic = FF2_MAGIC};
_Noreturn void ff2_fail(unsigned code)
{ fprintf(stderr, "FF2 recording failed: %u\n", code); _Exit(91); }
void ff2_sink(const struct ff2_event *e)
{
    if (fwrite(e, sizeof *e, 1, file) != 1) ff2_fail(118);
    header.count++;
}
static void finish(void)
{
    ff2_finish();
    if (fseek(file, 0, SEEK_SET) || fwrite(&header, sizeof header, 1, file) != 1 ||
        fclose(file)) ff2_fail(119);
}
static void init(void)
{
    pthread_mutexattr_t attr;
    const char *path = getenv("FFPOOL_TRACE");
    if (!path) ff2_fail(120);
    if (pthread_mutexattr_init(&attr) || pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) ||
        pthread_mutex_init(&mutex, &attr)) ff2_fail(121);
    pthread_mutexattr_destroy(&attr);
    file = fopen(path, "wb");
    if (!file || fwrite(&header, sizeof header, 1, file) != 1 || atexit(finish)) ff2_fail(122);
}
void ff2_lock(void)
{ if (pthread_once(&once, init) || pthread_mutex_lock(&mutex)) ff2_fail(123); }
void ff2_unlock(int *unused)
{ (void)unused; if (pthread_mutex_unlock(&mutex)) ff2_fail(124); }
