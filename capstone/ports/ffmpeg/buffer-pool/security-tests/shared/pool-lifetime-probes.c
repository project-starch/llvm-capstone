/* Functional lifetime probes against the actual ported pool APIs. Faulting
 * loads/stores have global labels so the runner can verify the fault PC.
 * A paired live access and successful setup marker precede every stale access.
 */
#include "libavutil/buffer.h"
#include "libavutil/mem.h"
#include "libavutil/refstruct.h"
#include "trace.h"
#ifdef FFPOOL_DOMAIN
#include "../../src/capstone-domain/node-snapshots.h"
#include "../../src/capstone-domain/alias-scatter.h"
#endif
#ifdef FFPOOL_CHERI
#include <stdio.h>
#endif

#define CHECK(c, n) do { if (!(c)) ff2_fail(n); } while (0)
static volatile unsigned char *held;
#ifdef FFPOOL_DOMAIN
static volatile unsigned char *volatile scattered_global;
static volatile unsigned long scattered_canary;
struct scattered_holder {
    volatile unsigned char *volatile alias;
    volatile unsigned long canary;
};
struct scattered_link {
    struct scattered_link *volatile next;
    volatile unsigned char *volatile alias;
    volatile unsigned long canary;
};
#endif
static void checkpoint(unsigned long round)
{
#ifdef FFPOOL_DOMAIN
    ff2_node_snapshot(round);
#else
    (void)round;
#endif
}
static void mark(unsigned id)
{
#ifdef FFPOOL_DOMAIN
    extern void ff2_probe_read(void), ff2_probe_write(void);
    extern void ff2_probe_register_read(void), ff2_probe_register_write(void);
    unsigned long value = 0xff25000000000000UL | id;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %3, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %4, x0\n"
                     : : "r"(value), "r"(ff2_probe_read), "r"(ff2_probe_write),
                         "r"(ff2_probe_register_read),
                         "r"(ff2_probe_register_write) : "memory");
#elif defined(FFPOOL_CHERI)
    printf("FF2_PROBE case=%u ready\n", id);
    fflush(stdout);
#else
    (void)id;
#endif
}
__attribute__((noinline)) static unsigned read_probe(const volatile unsigned char *p)
{
#ifdef FFPOOL_DOMAIN
    unsigned long value;
    __asm__ volatile(".globl ff2_probe_read\nff2_probe_read:\nlbu %0, 0(%1)\n" : "=r"(value) : "r"(p) : "memory");
    return value;
#else
    return *p;
#endif
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p)
{
#ifdef FFPOOL_DOMAIN
    unsigned long value = 93;
    __asm__ volatile(".globl ff2_probe_write\nff2_probe_write:\nsb %0, 0(%1)\n" : : "r"(value), "r"(p) : "memory");
#else
    *p = 93;
#endif
}
struct object { AVBufferRef *child; unsigned long magic; };
static unsigned inits, resets, frees;
static int object_init(AVRefStructOpaque unused, void *p)
{ (void)unused; ((struct object *)p)->magic = 0x13579; inits++; return 0; }
static void object_reset(AVRefStructOpaque unused, void *p)
{
    (void)unused;
    struct object *o = p;
    CHECK(o->magic == 0x13579, 410);
    av_buffer_unref(&o->child); resets++;
}
static void object_free(AVRefStructOpaque unused, void *p)
{ (void)unused; CHECK(((struct object *)p)->magic == 0x13579, 411); frees++; }

static void controls(void)
{
    AVBufferPool *bp = av_buffer_pool_init(64, NULL);
    AVBufferRef *a = av_buffer_pool_get(bp), *b = av_buffer_pool_get(bp);
    AVBufferRef *alias = av_buffer_ref(a);
    held = a->data; held[0] = 17; b->data[0] = 31;
    uintptr_t address = (uintptr_t)a->data;
    av_buffer_unref(&a);
    CHECK(read_probe(held) == 17, 401); /* another reference keeps it alive */
    av_buffer_unref(&alias);
    a = av_buffer_pool_get(bp);
    CHECK((uintptr_t)a->data == address && b->data[0] == 31, 402);
    a->data[0] = 42;
    av_buffer_pool_uninit(&bp);
    CHECK(a->data[0] == 42 && b->data[0] == 31, 403); /* deferred pool close */
    av_buffer_unref(&a); av_buffer_unref(&b); held = NULL;

    bp = av_buffer_pool_init(64, NULL);
    AVRefStructPool *rp = av_refstruct_pool_alloc_ext(sizeof(struct object), 0, NULL,
                                                       object_init, object_reset, object_free, NULL);
    struct object *r = av_refstruct_pool_get(rp);
    struct object *other = av_refstruct_pool_get(rp);
    r->child = av_buffer_pool_get(bp); r->child->data[0] = 51;
    struct object *ra = av_refstruct_ref(r);
    address = (uintptr_t)r;
    av_refstruct_unref(&r);
    CHECK(ra->magic == 0x13579 && ra->child->data[0] == 51 && resets == 0, 404);
    av_refstruct_unref(&ra);
    CHECK(resets == 1 && other->magic == 0x13579, 405);
    r = av_refstruct_pool_get(rp);
    CHECK((uintptr_t)r == address && r->magic == 0x13579 && !r->child && inits == 2, 406);
    av_refstruct_unref(&rp);
    CHECK(r->magic == 0x13579 && other->magic == 0x13579, 407);
    av_refstruct_unref(&r); av_refstruct_unref(&other);
    av_buffer_pool_uninit(&bp);
    CHECK(resets == 3 && frees == 2, 408);
    mark(0);
}

void ff2_security_run(unsigned test, unsigned long rounds, unsigned mode)
{
    (void)mode;
    if (!test) { controls(); return; }
    if (test == 8) {
        AVBufferPool *bp = av_buffer_pool_init(64, NULL);
        AVRefStructPool *rp = av_refstruct_pool_alloc_ext(sizeof(struct object), 0, NULL,
                                                           object_init, object_reset, object_free, NULL);
        struct object *o = av_refstruct_pool_get(rp);
        o->child = av_buffer_pool_get(bp); held = o->child->data; held[0] = 27;
        CHECK(read_probe(held) == 27, 420);
        av_refstruct_unref(&o); CHECK(resets == 1, 421); mark(test);
        (void)read_probe(held);
        av_refstruct_unref(&rp); av_buffer_pool_uninit(&bp); held = NULL;
        return;
    }
    if (test <= 4 || test == 9 || test == 11 || test == 12 || test == 13) {
        AVBufferPool *bp = av_buffer_pool_init(64, NULL);
        AVBufferRef *a = av_buffer_pool_get(bp), *sibling = av_buffer_pool_get(bp);
        held = a->data; held[0] = 17; sibling->data[0] = 41;
        CHECK(read_probe(held) == 17, 422);
        uintptr_t address = (uintptr_t)a->data;
        if (test == 11) { mark(test); (void)read_probe(held + 64); }
        else {
            av_buffer_unref(&a);
            if (test >= 3 && test != 9) {
                unsigned long n = (test == 12 || test == 13) ? rounds : 1;
                CHECK(n > 0 && n <= 3000000, 423);
                for (unsigned long i = 0; i < n; i++) {
                    a = av_buffer_pool_get(bp);
                    CHECK((uintptr_t)a->data == address, 424);
                    a->data[0] = 59; CHECK(a->data[0] == 59 && sibling->data[0] == 41, 425);
                    if ((test == 12 || test == 13) &&
                        (i == 0 || (i + 1) % 10000 == 0 || i + 1 == n))
                        checkpoint(i + 1);
                    if (i + 1 < n) av_buffer_unref(&a);
                }
            }
            if (test == 9) av_buffer_pool_uninit(&bp);
            CHECK(sibling->data[0] == 41, 426);
            mark(test);
            if (test == 2 || test == 4) write_probe(held);
            else if (test != 13) (void)read_probe(held);
        }
        av_buffer_unref(&a); av_buffer_unref(&sibling); av_buffer_pool_uninit(&bp); held = NULL;
        return;
    }
    if (test == 5 || test == 6 || test == 7 || test == 10) {
        AVRefStructPool *rp = av_refstruct_pool_alloc(64, 0);
        void *a = av_refstruct_pool_get(rp), *sibling = av_refstruct_pool_get(rp);
        held = a; held[0] = 19; ((unsigned char *)sibling)[0] = 43;
        CHECK(read_probe(held) == 19, 427);
        uintptr_t address = (uintptr_t)a;
        if (test == 10) { mark(test); (void)read_probe(held - 1); }
        else {
            av_refstruct_unref(&a);
            if (test == 6 || test == 7) {
                a = av_refstruct_pool_get(rp);
                CHECK((uintptr_t)a == address, 428);
                ((unsigned char *)a)[0] = 61;
                CHECK(((unsigned char *)a)[0] == 61 && ((unsigned char *)sibling)[0] == 43, 429);
            }
            mark(test);
            if (test == 6) write_probe(held);
            else if (test == 7) {
                void *stale = (void *)held;
                av_refstruct_unref(&stale); /* must reject before decrementing current refcount */
                a = NULL; /* baseline accepted the stale identity */
            } else (void)read_probe(held);
        }
        av_refstruct_unref(&a); av_refstruct_unref(&sibling); av_refstruct_unref(&rp); held = NULL;
        return;
    }
#ifdef FFPOOL_DOMAIN
    if (test >= 14 && test <= 35) {
        CHECK(mode == 0 || mode == 2, 443);
        unsigned control = test == 14 || test == 35;
        unsigned reuse = test == 14 || (test >= 25 && test <= 34);
        unsigned site = control ? 0 : ((test - 15) % 10) / 2;
        unsigned writing = control ? 0 : (test - 15) % 2;
        struct ff2_alias_scatter state = {0};
        struct scattered_holder *heap = av_mallocz(sizeof(*heap));
        struct scattered_link *head = av_mallocz(sizeof(*head));
        struct scattered_link *tail = av_mallocz(sizeof(*tail));
        CHECK(heap && head && tail, 431);
        head->next = tail;
        heap->canary = head->canary = tail->canary = scattered_canary = 71;

        ff2_alias_scatter_setup(&state);
        CHECK(state.child_alias && state.sibling_alias, 432);
        state.child_alias[0] = 17;
        state.sibling_alias[64] = 41;

        /* All holders are outside the parent subtree. The final holder uses
         * storage owned by an independent sibling pool. */
        scattered_global = state.child_alias;
        heap->alias = state.child_alias;
        tail->alias = state.child_alias;
        volatile unsigned char *volatile *other_pool =
            (volatile unsigned char *volatile *)state.sibling_alias;
        *other_pool = state.child_alias;
        volatile unsigned char *register_alias = state.child_alias;

        CHECK(read_probe(scattered_global) == 17, 433);
        CHECK(read_probe(heap->alias) == 17, 434);
        CHECK(read_probe(head->next->alias) == 17, 435);
        CHECK(read_probe(*other_pool) == 17, 436);
        CHECK(read_probe(register_alias) == 17 && state.sibling_alias[64] == 41, 437);

        if (!control && site == 4) {
            mark(test);
            CHECK(ff2_alias_scatter_register_span(register_alias,
                &state.parent_handle, reuse, writing, mode, state.sibling_alias,
                0xff26000000000000UL | test, state.child_address) == 0, 444);
            av_free(heap); av_free(head); av_free(tail);
            return;
        }

        ff2_alias_scatter_transition(&state, mode, reuse);
        if (reuse) {
            state.new_alias[0] = 61;
            CHECK(state.new_alias[0] == 61, 438);
        }
        CHECK(state.sibling_alias[64] == 41, 439);
        state.sibling_alias[65] = 42;
        CHECK(state.sibling_alias[65] == 42, 440);
        CHECK(heap->canary == 71 && head->canary == 71 &&
              tail->canary == 71 && scattered_canary == 71 &&
              head->next == tail, 441);
        heap->canary = 72;
        CHECK(heap->canary == 72, 442);
        if (control) {
            mark(test);
            av_free(heap); av_free(head); av_free(tail);
            return;
        }

        volatile unsigned char *selected =
            site == 0 ? scattered_global :
            site == 1 ? heap->alias :
            site == 2 ? head->next->alias : *other_pool;
        mark(test);
        if (writing) {
            write_probe(selected);
            CHECK(read_probe(selected) == 93, 445);
        } else {
            CHECK(read_probe(selected) == (reuse ? 61 : 17), 446);
        }
        av_free(heap); av_free(head); av_free(tail);
        return;
    }
#endif
    if (test == 36) {
        /* af_join's dedup bound (upstream 461fb22053). The tracking test
         * compares the search index against the CHANNEL index instead of
         * nb_buffers, so once channel 1 shares channel 0's buffer, channel 2's
         * genuinely new buffer ends the search at j == nb_buffers != i and is
         * never tracked -- so no reference is taken for it, and it returns to
         * the pool while the output still names it. */
        AVBufferPool *bp = av_buffer_pool_init(64, NULL);
        AVBufferRef *in0 = av_buffer_pool_get(bp), *in1 = av_buffer_pool_get(bp);
        CHECK(in0 && in1, 450);
        in0->data[0] = 17; in0->data[32] = 19; in1->data[0] = 23;
        AVBufferRef *tracked[3];
        volatile unsigned char *chan[3];
        unsigned nb = 0;
        for (unsigned i = 0; i < 3; i++) {
            AVBufferRef *cur = i < 2 ? in0 : in1;
            unsigned j;
            chan[i] = cur->data + (i == 1 ? 32 : 0);
            for (j = 0; j < nb; j++)
                if (tracked[j]->buffer == cur->buffer)
                    break;
            if (j == i)
                tracked[nb++] = cur;
        }
        CHECK(nb == 1, 451); /* channel 2's buffer was dropped */
        AVBufferRef *out = av_buffer_ref(tracked[0]);
        CHECK(out, 452);
        held = chan[2];
        CHECK(read_probe(held) == 23, 453);
        uintptr_t address = (uintptr_t)held;
        av_buffer_unref(&in0); /* the output reference keeps this one alive */
        av_buffer_unref(&in1); /* nothing references this one */
        AVBufferRef *next = av_buffer_pool_get(bp);
        CHECK(next && (uintptr_t)next->data == address, 454);
        next->data[0] = 61;
        CHECK(out->data[0] == 17 && out->data[32] == 19, 455);
        mark(test);
        CHECK(read_probe(held) == 61, 456);
        av_buffer_unref(&next); av_buffer_unref(&out);
        av_buffer_pool_uninit(&bp); held = NULL;
        return;
    }
    ff2_fail(430);
}
