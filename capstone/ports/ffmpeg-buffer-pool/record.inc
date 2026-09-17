/* Included only in the temporary native FFmpeg source tree.
 * Serialize whole pool operations (including last-reference callbacks) so
 * recording order agrees with the pool's free-list linearization order.
 * FFmpeg's native atomics and pthread support remain enabled. This lock can
 * change scheduling: this is a functional memory trace, not a timing study.
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "replay-format.h"

static pthread_mutex_t ff_record_mutex;
static pthread_once_t ff_record_once = PTHREAD_ONCE_INIT;
static FILE *ff_record_file;
static struct ff_header ff_record_header = { FFTRACE_MAGIC, 0, 0, 0 };
static uint64_t ff_record_allocs, ff_record_live, ff_record_retained;
static unsigned ff_record_np, ff_record_nb;
static struct {
    AVBufferPool *ptr;
    uint64_t id;
    int closed;
} ff_record_pools[1024];
static struct {
    uint8_t *data;
    uint64_t pool, size, lease, freed_at;
    int alive, live;
} ff_record_buffers[8192];

static void ff_record_fail(void) { fputs("FFTRACE recorder failed\n", stderr); _Exit(91); }
static void ff_record_emit(struct ff_event e)
{
    e.allocations = ff_record_allocs;
    e.live_bytes = ff_record_live;
    e.retained_bytes = ff_record_retained;
    if (fwrite(&e, sizeof e, 1, ff_record_file) != 1) ff_record_fail();
    ff_record_header.count++;
}
static void ff_record_finish(void)
{
    for (unsigned i = 1; i <= ff_record_np; i++)
        if (!ff_record_pools[i].closed) ff_record_fail();
    if (ff_record_live || ff_record_retained) ff_record_fail();
    ff_record_emit((struct ff_event){.op=FF_END});
    if (fseek(ff_record_file, 0, SEEK_SET) ||
        fwrite(&ff_record_header, sizeof ff_record_header, 1, ff_record_file) != 1 ||
        fclose(ff_record_file)) ff_record_fail();
}
static void ff_record_init(void)
{
    pthread_mutexattr_t attr;
    const char *path = getenv("FFPOOL_TRACE");
    if (!path) ff_record_fail();
    if (pthread_mutexattr_init(&attr) ||
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) ||
        pthread_mutex_init(&ff_record_mutex, &attr)) ff_record_fail();
    pthread_mutexattr_destroy(&attr);
    ff_record_file = fopen(path, "wb");
    if (!ff_record_file || fwrite(&ff_record_header, sizeof ff_record_header, 1,
                                 ff_record_file) != 1 || atexit(ff_record_finish))
        ff_record_fail();
}
static void ff_record_lock(void)
{
    if (pthread_once(&ff_record_once, ff_record_init) ||
        pthread_mutex_lock(&ff_record_mutex)) ff_record_fail();
}
static void ff_record_unlock(int *unused)
{
    (void)unused;
    if (pthread_mutex_unlock(&ff_record_mutex)) ff_record_fail();
}
#define FF_RECORD_GUARD ff_record_lock(); \
    int ff_record_guard __attribute__((cleanup(ff_record_unlock))) = 0

static unsigned ff_record_pool(AVBufferPool *p)
{
    for (unsigned i = ff_record_np; i; i--)
        if (ff_record_pools[i].ptr == p && !ff_record_pools[i].closed) return i;
    /* A final reference may be returned after owner CLOSE. */
    for (unsigned i = ff_record_np; i; i--)
        if (ff_record_pools[i].ptr == p) return i;
    ff_record_fail(); return 0;
}
static unsigned ff_record_buffer(unsigned p, uint8_t *data)
{
    for (unsigned i = 1; i <= ff_record_nb; i++)
        if (ff_record_buffers[i].alive && ff_record_buffers[i].pool == p &&
            ff_record_buffers[i].data == data) return i;
    ff_record_fail(); return 0;
}
static void ff_record_create(AVBufferPool *p)
{
    unsigned i = ++ff_record_np;
    if (!p || i >= 1024 || p->alloc2 ||
        (p->alloc != av_buffer_alloc && p->alloc != av_buffer_allocz)) ff_record_fail();
    ff_record_pools[i].ptr = p;
    ff_record_pools[i].id = i;
    ff_record_emit((struct ff_event){.op=FF_CREATE, .pool=i, .size=p->size,
                                   .aux=p->alloc == av_buffer_allocz});
}
static void ff_record_get(AVBufferPool *p, AVBufferRef *r, int reuse)
{
    unsigned pi = ff_record_pool(p), bi;
    if (!r || ff_record_pools[pi].closed) ff_record_fail();
    if (reuse) {
        bi = ff_record_buffer(pi, r->buffer->data);
        if (ff_record_buffers[bi].live) ff_record_fail();
    } else {
        bi = ++ff_record_nb;
        if (bi >= 8192) ff_record_fail();
        ff_record_buffers[bi].data = r->buffer->data;
        ff_record_buffers[bi].pool = pi;
        ff_record_buffers[bi].size = p->size;
        ff_record_buffers[bi].alive = 1;
        ff_record_retained += p->size;
    }
    ff_record_allocs++;
    ff_record_live += p->size;
    ff_record_buffers[bi].live = 1;
    ff_record_buffers[bi].lease = ff_record_allocs;
    ff_record_emit((struct ff_event){.op=FF_GET, .pool=pi, .lease=ff_record_allocs,
        .backing=bi, .size=p->size,
        .gap=reuse ? ff_record_allocs - ff_record_buffers[bi].freed_at : 0});
}
static struct ff_event ff_record_return(AVBufferPool *p, uint8_t *data)
{
    unsigned pi = ff_record_pool(p), bi = ff_record_buffer(pi, data);
    if (!ff_record_buffers[bi].live) ff_record_fail();
    ff_record_buffers[bi].live = 0;
    ff_record_buffers[bi].freed_at = ff_record_allocs;
    ff_record_live -= ff_record_buffers[bi].size;
    return (struct ff_event){.op=FF_RETURN, .pool=pi,
        .lease=ff_record_buffers[bi].lease, .backing=bi, .size=ff_record_buffers[bi].size};
}
static struct ff_event ff_record_close(AVBufferPool *p)
{
    unsigned pi = ff_record_pool(p);
    if (ff_record_pools[pi].closed) ff_record_fail();
    ff_record_pools[pi].closed = 1;
    return (struct ff_event){.op=FF_CLOSE, .pool=pi};
}
static void ff_record_drop(AVBufferPool *p, uint8_t *data)
{
    unsigned pi = ff_record_pool(p), bi = ff_record_buffer(pi, data);
    if (ff_record_buffers[bi].live) ff_record_fail();
    ff_record_buffers[bi].alive = 0;
    ff_record_retained -= ff_record_buffers[bi].size;
}
