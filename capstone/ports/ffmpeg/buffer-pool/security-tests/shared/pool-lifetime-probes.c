/* Functional lifetime probes against the actual ported pool APIs. Faulting
 * loads/stores have global labels so the runner can verify the fault PC.
 * A paired live access and successful setup marker precede every stale access.
 */
#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
#include "trace.h"

#define CHECK(c, n) do { if (!(c)) ff2_fail(n); } while (0)
static volatile unsigned char *held;
static void mark(unsigned id)
{
#ifdef FFPOOL_DOMAIN
    extern void ff2_probe_read(void), ff2_probe_write(void);
    unsigned long value = 0xff25000000000000UL | id;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                     ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                     : : "r"(value), "r"(ff2_probe_read), "r"(ff2_probe_write) : "memory");
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

void ff2_security_run(unsigned test, unsigned long rounds)
{
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
                CHECK(n > 0 && n <= 1000000, 423);
                for (unsigned long i = 0; i < n; i++) {
                    a = av_buffer_pool_get(bp);
                    CHECK((uintptr_t)a->data == address, 424);
                    a->data[0] = 59; CHECK(a->data[0] == 59 && sibling->data[0] == 41, 425);
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
    ff2_fail(430);
}
