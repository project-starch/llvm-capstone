/* Compile the same public-API probe against upstream libavutil, isolated
 * buffer.c, and a Capstone domain. This is a serial semantic fixture, not
 * an application recording or a Sublet security result.
 */
#include <stdint.h>
#include <string.h>
#include "libavutil/buffer.h"

#ifndef FFPOOL_DOMAIN
#include <stdio.h>
#endif

struct counters {
    unsigned allocated, released, destroyed;
};

static void release_payload(void *opaque, uint8_t *data)
{
    struct counters *c = opaque;
    void av_free(void *);
    c->released++;
    av_free(data);
}

static AVBufferRef *allocate_payload(void *opaque, size_t size)
{
    struct counters *c = opaque;
    void *av_malloc(size_t);
    void av_free(void *);
    uint8_t *data = av_malloc(size);
    AVBufferRef *ref;
    if (!data)
        return 0;
    ref = av_buffer_create(data, size, release_payload, opaque, 0);
    if (!ref) {
        av_free(data);
        return 0;
    }
    c->allocated++;
    return ref;
}

static void destroy_pool(void *opaque)
{
    struct counters *c = opaque;
    c->destroyed++;
}

/* Failure numbers identify semantic checkpoints without depending on stdio.
 * Retained integer addresses below are only reuse oracles, never converted
 * back to pointers. The probe does not access data after its lifetime ends.
 */
#define CHECK(cond, code) do { if (!(cond)) return (code); } while (0)

static unsigned exercise_pool(void)
{
    struct counters c = {0, 0, 0};
    AVBufferPool *pool = av_buffer_pool_init2(256, &c, allocate_payload,
                                             destroy_pool);
    AVBufferRef *a, *alias, *b, *replacement;
    uintptr_t address;

    CHECK(pool, 101);
    a = av_buffer_pool_get(pool);
    CHECK(a && a->size == 256, 102);
    memset(a->data, 0x31, a->size);
    address = (uintptr_t)a->data;
    alias = av_buffer_ref(a);
    CHECK(alias && av_buffer_get_ref_count(a) == 2, 103);
    CHECK(!av_buffer_is_writable(a), 104);
    av_buffer_unref(&a);
    CHECK(!a && av_buffer_get_ref_count(alias) == 1, 105);
#ifdef FFPOOL_FORCE_FAILURE
    /* Negative control: the payload oracle must detect a corrupted byte. */
    alias->data[255] = 0;
#endif
    CHECK(alias->data[0] == 0x31 && alias->data[255] == 0x31, 106);

    /* Dropping one reference has NOT returned its buffer to the pool. */
    b = av_buffer_pool_get(pool);
    CHECK(b && (uintptr_t)b->data != address && c.allocated == 2, 107);
    av_buffer_unref(&b);
    CHECK(c.released == 0, 108);
    av_buffer_unref(&alias);
    CHECK(!alias && c.released == 0, 109);

    /* The last return is the first reused entry in this serial fixture. */
    replacement = av_buffer_pool_get(pool);
    CHECK(replacement && (uintptr_t)replacement->data == address, 110);
    CHECK(c.allocated == 2, 111);
    CHECK(av_buffer_pool_buffer_get_opaque(replacement) == &c, 112);
    memset(replacement->data, 0x72, replacement->size);

    /* uninit flushes the idle entry, but the issued replacement survives. */
    av_buffer_pool_uninit(&pool);
    CHECK(!pool && c.released == 1 && c.destroyed == 0, 113);
    CHECK(replacement->data[0] == 0x72 && replacement->data[255] == 0x72, 114);
    alias = av_buffer_ref(replacement);
    CHECK(alias, 115);
    av_buffer_unref(&replacement);
    CHECK(c.released == 1 && c.destroyed == 0, 116);
    CHECK(alias->data[0] == 0x72, 117);
    av_buffer_unref(&alias);
    CHECK(c.released == 2 && c.destroyed == 1, 118);
    return 0;
}

static unsigned exercise_detach(void)
{
    struct counters c = {0, 0, 0};
    AVBufferPool *pool = av_buffer_pool_init2(128, &c, allocate_payload,
                                             destroy_pool);
    AVBufferRef *a, *alias, *next;
    uintptr_t address;
    CHECK(pool, 201);
    a = av_buffer_pool_get(pool);
    CHECK(a, 202);
    memset(a->data, 0x45, a->size);
    address = (uintptr_t)a->data;
    alias = av_buffer_ref(a);
    CHECK(alias, 203);

    /* Copy-on-write changes which backing buffer this reference names. */
    CHECK(av_buffer_make_writable(&a) == 0, 204);
    CHECK((uintptr_t)a->data != address && a->data[127] == 0x45, 205);
    a->data[0] = 0x66;
    CHECK(alias->data[0] == 0x45, 206);
    CHECK(av_buffer_realloc(&a, 512) == 0, 207);
    CHECK(a->size == 512 && a->data[0] == 0x66 && a->data[127] == 0x45, 208);
    av_buffer_unref(&alias);
    next = av_buffer_pool_get(pool);
    CHECK(next && (uintptr_t)next->data == address && c.allocated == 1, 209);
    CHECK(av_buffer_pool_buffer_get_opaque(next) == &c, 210);
    av_buffer_unref(&next);
    av_buffer_pool_uninit(&pool);
    CHECK(c.released == 1 && c.destroyed == 1, 211);
    CHECK(a->data[0] == 0x66 && a->data[127] == 0x45, 212);
    av_buffer_unref(&a);
    return 0;
}

static unsigned probe(void)
{
    unsigned result = exercise_pool();
    if (!result)
        result = exercise_detach();
    return result;
}

#ifdef FFPOOL_DOMAIN
/* call_dom's return slot is an unsigned long, including on the host side. */
void domain_main(unsigned long *result, unsigned func)
{
    unsigned code;
    (void)func;
    code = probe();
    *result = code ? code : 42042;
}
#else
int main(void)
{
    unsigned code = probe();
    printf("ffpool status=%u result=%s\n", code, code ? "FAIL" : "PASS");
    return code ? 1 : 0;
}
#endif
