/* Serial replay against unmodified upstream buffer.c. Ref/unref traffic is
 * collapsed to GET and the final RETURN: this tests pool behavior, not refcount
 * operation cost. Never consult recorded backing choices or memory counters.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "libavutil/buffer.h"
#include "libavutil/mem.h"
#include "replay-format.h"

void ff_memory_init(void *, size_t);
size_t ff_memory_used(void);
static struct pool_state { AVBufferPool *pool; uint64_t id, size, zero, closed; } pools[1024];
static struct backing_state {
    uint8_t *data;
    uint64_t pool, size, freed_at;
    unsigned alive, live, used;
} backing[8192];
static struct lease_state { AVBufferRef *ref; uint64_t id, backing, pool; } leases[1024];
static uint64_t allocations, live_bytes, retained_bytes, nbacking, npools;
static unsigned callback_error;

static void backing_free(void *opaque, uint8_t *data)
{
    struct backing_state *b = opaque;
    if (!b->alive || b->live || b->data != data || retained_bytes < b->size)
        callback_error = 1;
    b->alive = 0;
    retained_bytes -= b->size;
    av_free(data);
}
static AVBufferRef *backing_alloc(void *opaque, size_t size)
{
    struct pool_state *p = opaque;
    if (nbacking + 1 >= 8192 || size != p->size) { callback_error = 2; return NULL; }
    uint8_t *data = p->zero ? av_mallocz(size) : av_malloc(size);
    if (!data) return NULL;
    struct backing_state *b = &backing[++nbacking];
    b->data = data; b->pool = p->id; b->size = size; b->alive = 1;
    retained_bytes += size;
    AVBufferRef *r = av_buffer_create(data, size, backing_free, b, 0);
    if (!r) { backing_free(b, data); return NULL; }
    return r;
}

/* Error code + completed-event count remain visible even on a rejected trace. */
static unsigned replay(const struct ff_header *input, struct ff_header *output,
                       void *memory)
{
    const struct ff_event *events = (const struct ff_event *)(input + 1);
    struct ff_event *observed = (struct ff_event *)(output + 1);
    *output = (struct ff_header){FFTRACE_MAGIC, 0, 1, 0};
    if (input->magic != FFTRACE_MAGIC || input->status || !input->count ||
        input->count > (FFTRACE_BYTES - sizeof *input) / sizeof *events)
        return 10;
    ff_memory_init(memory, FFARENA_BYTES);
    for (uint64_t i = 0; i < input->count; i++) {
        const struct ff_event *e = &events[i];
        struct ff_event o = {0};
        o.op = e->op; o.pool = e->pool;
        if (e->pool >= 1024) return 11;
        struct pool_state *p = &pools[e->pool];
        switch (e->op) {
        case FF_CREATE:
            if (!e->pool || e->pool != npools + 1 || p->id || !e->size ||
                e->size > FFARENA_BYTES || e->aux > 1) return 12;
            p->id = e->pool; p->size = e->size; p->zero = e->aux;
            p->pool = av_buffer_pool_init2(e->size, p, backing_alloc, NULL);
            if (!p->pool) return 13;
            npools++;
            o.size = p->size; o.aux = p->zero;
            break;
        case FF_GET: {
            if (!p->pool || p->closed || e->size != p->size ||
                e->lease != allocations + 1) return 14;
            unsigned li;
            for (li = 0; li < 1024 && leases[li].ref; li++) {}
            if (li == 1024) return 15;
            AVBufferRef *r = av_buffer_pool_get(p->pool);
            if (!r || r->size != p->size) return 16;
            uint64_t bi;
            for (bi = 1; bi <= nbacking; bi++)
                if (backing[bi].alive && backing[bi].pool == p->id &&
                    backing[bi].data == r->data) break;
            if (bi > nbacking || backing[bi].live) return 17;
            struct backing_state *b = &backing[bi];
            allocations++;
            o.gap = b->used ? allocations - b->freed_at : 0;
            b->used = 1; b->live = 1;
            leases[li].ref = r; leases[li].id = e->lease;
            leases[li].backing = bi; leases[li].pool = p->id;
            live_bytes += r->size;
            o.lease = e->lease; o.backing = bi; o.size = r->size;
            /* Exercise payload access without pretending to replay decoding. */
            r->data[0] = (uint8_t)e->lease;
            r->data[r->size - 1] = (uint8_t)(e->lease >> 8);
            break;
        }
        case FF_RETURN: {
            unsigned li;
            for (li = 0; li < 1024; li++)
                if (leases[li].ref && leases[li].id == e->lease) break;
            if (li == 1024 || leases[li].pool != e->pool) return 18;
            struct lease_state *l = &leases[li];
            struct backing_state *b = &backing[l->backing];
            if (!b->live || e->size != b->size ||
                l->ref->data[0] != (uint8_t)e->lease ||
                l->ref->data[b->size - 1] != (uint8_t)(e->lease >> 8)) return 19;
            o.lease = l->id; o.backing = l->backing; o.size = b->size;
            b->live = 0; b->freed_at = allocations;
            live_bytes -= b->size;
            av_buffer_unref(&l->ref);
            break;
        }
        case FF_CLOSE:
            if (!p->pool || p->closed) return 20;
            p->closed = 1;
            av_buffer_pool_uninit(&p->pool);
            break;
        case FF_END:
            if (i + 1 != input->count || live_bytes || retained_bytes) return 21;
            for (unsigned j = 1; j <= npools; j++)
                if (!pools[j].closed) return 22;
            break;
        default: return 23;
        }
        if (callback_error) return 30 + callback_error;
        o.allocations = allocations;
        o.live_bytes = live_bytes; o.retained_bytes = retained_bytes;
        observed[i] = o;
        output->count = i + 1;
    }
    if (events[input->count - 1].op != FF_END) return 24;
    output->arena_used = ff_memory_used();
    output->status = 0;
    return 0;
}

#ifdef FFPOOL_DOMAIN
static struct ff_header *report;
static const struct ff_header *trace;
static void *arena_region;
static unsigned shares;
void domain_main(unsigned long *result, unsigned func)
{
    if (func == 1) {
        switch (shares++) {
        case 0: report = (struct ff_header *)result; break;
        case 1: arena_region = result; break;
        case 2: trace = (const struct ff_header *)result; break;
        }
        return;
    }
    if (shares != 3) { *result = 99; return; }
    unsigned status = replay(trace, report, arena_region);
    report->status = status;
    *result = status ? status : 42043;
}
#else
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    struct ff_header *input = calloc(1, FFTRACE_BYTES);
    struct ff_header *output = calloc(1, FFREPORT_BYTES);
    void *memory = aligned_alloc(64, FFARENA_BYTES);
    FILE *f = fopen(argv[1], "rb");
    if (!input || !output || !memory || !f) return 2;
    size_t n = fread(input, 1, FFTRACE_BYTES, f);
    if (ferror(f) || fgetc(f) != EOF || n < sizeof *input ||
        input->count > (FFTRACE_BYTES - sizeof *input) / sizeof(struct ff_event) ||
        n != sizeof *input + input->count * sizeof(struct ff_event)) return 3;
    fclose(f);
    unsigned status = replay(input, output, memory);
    output->status = status;
    f = fopen(argv[2], "wb");
    if (!f) return 4;
    size_t bytes = sizeof *output + output->count * sizeof(struct ff_event);
    if (fwrite(output, 1, bytes, f) != bytes || fclose(f)) return 4;
    printf("FFREPLAY status=%u events=%llu arena=%llu\n", status,
           (unsigned long long)output->count, (unsigned long long)output->arena_used);
    free(input); free(output); free(memory);
    return status ? 1 : 0;
}
#endif
