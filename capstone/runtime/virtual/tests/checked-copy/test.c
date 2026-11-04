/* SPDX-License-Identifier: MIT */
/* Hosted adaptation of inline-io-contract.c's blocked copier, revoke/reuse,
 * pipe handshake and exact sentinel oracle. This consumes lane N's real
 * context double. Fault injection models __copy_*_inatomic's short result;
 * mmap/MAP_FIXED and private file COW below are actual host operations. */
#define _GNU_SOURCE
#include <capstone/checked-copy.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE 4096
#define SIZE (2 * PAGE)
static unsigned checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
    printf("FAIL line %u: %s\n", __LINE__, #x); } } while (0)
struct fixture {
    struct cn_engine *engine;
    struct cn_context *context;
    struct cn_owner owner;
    struct cn_capref ref;
    pthread_mutex_t exclusion;
    pthread_t worker;
    int go[2], ack[2], fd;
    unsigned char *memory, *replacement;
    cn_u64 generation;
    unsigned rights, type, holds, entered, drained, copies, faults, switches;
    int mode, retired, pending, fault_ready, fired, worker_started;
};
enum { LIVE, OBJECT, MAPPING, COW_OBJECT, COW_MAPPING, STUCK, PARTIAL };
static void handshake(int fd, int send)
{
    char byte = 'x';
    assert((send ? write(fd, &byte, 1) : read(fd, &byte, 1)) == 1);
}
static int validate(void *v, const struct cn_owner *o, const struct cn_translation *t)
{
    struct fixture *f = v;
    return o->namespace_id == f->owner.namespace_id &&
           t->namespace_id == o->namespace_id && t->namespace_epoch == o->namespace_epoch ? 0 : -EPERM;
}
static int authorize(void *v, const struct cn_owner *o, const struct cn_register *r,
                     cn_u64 offset, cn_u64 length, int write)
{
    struct fixture *f = v;
    (void)o;
    if (f->retired || r->payload[1] != f->generation) return CU_LIFETIME;
    if (f->type != 1) return CU_TYPE;
    if (!(f->rights & (write ? 2 : 4))) return CU_RIGHTS;
    if (offset > SIZE || length > SIZE - offset) return CU_BOUNDS;
    return 0;
}
static int step(void *v, struct cn_frame *fr, struct cn_event *e)
{
    (void)v; (void)fr;
    e->kind = CN_EVENT_QUANTUM;
    return 0;
}
static void *revoker(void *v)
{
    struct fixture *f = v;
    handshake(f->go[0], 0);
    pthread_mutex_lock(&f->exclusion);
    ++f->generation;
    if (f->mode == MAPPING || f->mode == COW_MAPPING) {
        f->retired = 1; /* logical munmap returns 0 immediately */
        if (f->holds) {
            f->pending = 1;
            errno = 0;
            void *p = mmap(f->memory, SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            assert(p == MAP_FAILED && errno == EEXIST); /* VA quarantine */
            f->replacement = mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            assert(f->replacement != MAP_FAILED && f->replacement != f->memory);
        } else {
            assert(!munmap(f->memory, SIZE));
            f->replacement = mmap(f->memory, SIZE, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
            assert(f->replacement == f->memory);
        }
        memset(f->replacement, 0xaa, SIZE);
    } else {
        /* The old object's second page is reused inside a still held arena.
         * For COW this store creates a private page; the file stays 0x11. */
        assert(!mprotect(f->memory, SIZE, PROT_READ | PROT_WRITE));
        memset(f->memory + PAGE, 0xaa, PAGE);
    }
    ++f->switches;
    pthread_mutex_unlock(&f->exclusion);
    handshake(f->ack[1], 1);
    return NULL;
}
static void switch_thread(struct fixture *f)
{
    if (f->fired || !f->worker_started) return;
    assert(!f->entered);
    f->fired = 1;
    handshake(f->go[1], 1);
    handshake(f->ack[0], 0);
}
static cn_u32 hold(void *v, const struct cn_capref *ref)
{
    struct fixture *f = v;
    int rc = cn_authorize(f->context, &f->owner, ref, 0, 0, 0);
    if (rc) return rc > 0 ? (cn_u32)rc : CU_REFUSED;
#ifndef CU_MUTATE_NO_HOLD
    ++f->holds;
#endif
    return 0;
}
static void release(void *v)
{
    struct fixture *f = v;
#ifndef CU_MUTATE_NO_HOLD
    assert(f->holds == 1);
    --f->holds;
#endif
    if (f->pending) {
        assert(!munmap(f->memory, SIZE));
        f->memory = NULL;
        f->pending = 0;
    }
}
static void enter(void *v)
{
    struct fixture *f = v;
    pthread_mutex_lock(&f->exclusion);
    assert(!f->entered);
    f->entered = 1; f->drained = 0;
}
static cn_u32 check(void *v, const struct cn_capref *ref, cn_u64 offset,
                    cn_u64 length, enum cu_direction dir, cn_u64 *address)
{
    struct fixture *f = v;
    struct cn_frame fr;
    assert(f->entered);
    int rc = cn_authorize(f->context, &f->owner, ref, offset, length, dir == CU_TO_DOMAIN);
#ifdef CU_MUTATE_SKIP_CHECK
    if (f->copies) rc = 0;
#endif
    if (rc) return rc > 0 ? (cn_u32)rc : rc == -EACCES ? CU_TAG : CU_REFUSED;
    assert(!cn_snapshot(f->context, &f->owner, &fr));
    *address = fr.x[ref->frame_slot].payload[0] + offset;
#ifdef CU_MUTATE_NO_EXCLUSION
    if (length && f->copies && f->mode == OBJECT && !f->fired) {
        f->entered = 0;
        pthread_mutex_unlock(&f->exclusion);
        switch_thread(f);
        pthread_mutex_lock(&f->exclusion);
        f->entered = 1;
    }
#endif
    return 0;
}
static cn_u64 copy(void *v, cn_u64 address, void *buffer, cn_u64 len, enum cu_direction dir)
{
    struct fixture *f = v;
    assert(f->entered && len <= PAGE && (address % PAGE) + len <= PAGE);
    ++f->copies;
    if (f->mode == STUCK ||
        ((f->mode == COW_OBJECT || f->mode == COW_MAPPING) && !f->fault_ready &&
         address >= (cn_u64)f->memory + PAGE)) return len;
    cn_u64 n = f->mode == PARTIAL && !f->fault_ready ? len / 2 : len;
    if (dir == CU_TO_DOMAIN) memcpy((void *)address, buffer, n);
    else memcpy(buffer, (void *)address, n);
    return len - n;
}
static void drain(void *v) { ((struct fixture *)v)->drained = 1; }
static void leave(void *v)
{
    struct fixture *f = v;
    assert(f->entered && f->drained);
    f->entered = 0;
    pthread_mutex_unlock(&f->exclusion);
}
static cn_u32 resolve(void *v, cn_u64 address, enum cu_direction dir)
{
    struct fixture *f = v;
    (void)address; (void)dir;
    assert(!f->entered);
    ++f->faults;
    switch_thread(f);
    f->fault_ready = 1;
#ifdef CU_MUTATE_DUMMY_STORE
    if (f->mode == COW_OBJECT) *(unsigned char *)address = 0;
#endif
    return 0;
}
static void between(void *v)
{
    struct fixture *f = v;
    #ifdef CU_MUTATE_NO_EXCLUSION
    if (f->mode == OBJECT) return;
#endif
    if (f->mode == OBJECT || f->mode == MAPPING) switch_thread(f);
}
static const struct cu_ops ops = {hold, release, enter, check, copy, drain, leave, resolve, between};
static void setup(struct fixture *f, int mode)
{
    memset(f, 0, sizeof(*f));
    f->fd = -1; f->mode = mode; f->generation = 1; f->rights = 6; f->type = 1;
    f->owner = (struct cn_owner){1, 2, 3, 4};
    assert(!pthread_mutex_init(&f->exclusion, NULL));
    if (mode == COW_OBJECT || mode == COW_MAPPING) {
        char name[] = "/tmp/capstone-copy-XXXXXX";
        f->fd = mkstemp(name); assert(f->fd >= 0); unlink(name);
        unsigned char initial[SIZE]; memset(initial, 0x11, SIZE);
        assert(write(f->fd, initial, SIZE) == SIZE);
        f->memory = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE, f->fd, 0);
    } else f->memory = mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(f->memory != MAP_FAILED);
    if (f->fd < 0) memset(f->memory, 0x11, SIZE);
    struct cn_frame fr = {.version=CN_ABI_VERSION, .bytes=sizeof(fr)};
    fr.translation = (struct cn_translation){.root_pa=4096, .translation_epoch=1,
        .address_space_id=2, .namespace_id=3, .namespace_epoch=4};
    fr.x[10].payload[0] = (cn_u64)f->memory;
    fr.x[10].payload[1] = f->generation;
    fr.x[10].flags = CN_REG_TAG;
    const struct cn_ops nops = {.validate_namespace=validate, .authorize=authorize, .step=step};
    assert(!cn_engine_create(CN_BACKEND_IOCTL, &nops, f, &f->engine));
    assert(!cn_create(f->engine, &f->owner, &fr, &f->context));
    assert(!cn_get_capref(f->context, &f->owner, 10, &f->ref));
    if (mode >= OBJECT && mode <= COW_MAPPING) {
        assert(!pipe(f->go) && !pipe(f->ack));
        f->worker_started = 1;
        assert(!pthread_create(&f->worker, NULL, revoker, f));
    }
}
static void cleanup(struct fixture *f)
{
    if (f->worker_started) {
        assert(f->fired);
        assert(!pthread_join(f->worker, NULL));
        close(f->go[0]); close(f->go[1]); close(f->ack[0]); close(f->ack[1]);
    }
    if (f->memory) assert(!munmap(f->memory, SIZE));
    if (f->replacement && f->replacement != f->memory) assert(!munmap(f->replacement, SIZE));
    if (f->fd >= 0) close(f->fd);
    cn_engine_free(f->engine);
    assert(!pthread_mutex_destroy(&f->exclusion));
}
static size_t changed(const unsigned char *p, size_t n)
{
    size_t result = 0;
    for (size_t i=0; i<n; ++i) result += p[i] != 0xaa;
    return result;
}
int main(void)
{
    unsigned char buffer[SIZE]; memset(buffer, 0x5c, SIZE);
    for (int mode=OBJECT; mode<=COW_MAPPING; ++mode) {
        struct fixture f; setup(&f, mode);
        struct cu_result r = cu_copy(&ops, &f, &f.ref, 0, buffer, SIZE, CU_TO_DOMAIN, PAGE);
        size_t bad = changed(f.replacement ? f.replacement : f.memory + PAGE,
                             f.replacement ? SIZE : PAGE);
        printf("reuse mode=%d done=%llu cause=%u unauthorized=%zu switches=%u faults=%u\n",
               mode, (unsigned long long)r.bytes_done, r.cause, bad, f.switches, f.faults);
        CHECK(r.bytes_done == PAGE && r.cause == CU_LIFETIME);
        CHECK(bad == 0 && f.switches == 1 && f.holds == 0 && !f.pending);
        if (mode >= COW_OBJECT) CHECK(f.faults == 1);
        if (f.fd >= 0) {
            unsigned char file[SIZE]; assert(pread(f.fd, file, SIZE, 0) == SIZE);
            for (unsigned i=0; i<SIZE; ++i) CHECK(file[i] == 0x11);
        }
        cleanup(&f);
    }
    struct fixture f; setup(&f, LIVE);
    struct cu_result r = cu_copy(&ops, &f, &f.ref, 0, buffer, SIZE, CU_TO_DOMAIN, PAGE);
    CHECK(!r.cause && r.bytes_done == SIZE && !memcmp(buffer, f.memory, SIZE));
    memset(buffer, 0, SIZE);
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, SIZE, CU_FROM_DOMAIN, PAGE);
    CHECK(!r.cause && r.bytes_done == SIZE && buffer[SIZE-1] == 0x5c);
    r = cu_copy(&ops, &f, &f.ref, SIZE, NULL, 0, CU_TO_DOMAIN, PAGE);
    CHECK(!r.cause && !r.bytes_done);
    r = cu_copy(&ops, &f, &f.ref, ~(cn_u64)0, buffer, 2, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_BOUNDS && !r.bytes_done);
    r = cu_copy(&ops, &f, &f.ref, SIZE, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_BOUNDS && !r.bytes_done);
    f.rights = 4;
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_RIGHTS && !r.bytes_done); f.rights = 6;
    f.type = 2;
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_TYPE && !r.bytes_done); f.type = 1;
    ++f.ref.context_epoch;
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_REFUSED && !r.bytes_done); --f.ref.context_epoch;
    ++f.ref.context_id;
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_REFUSED && !r.bytes_done); --f.ref.context_id;
    ++f.owner.namespace_id;
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_REFUSED && !r.bytes_done); --f.owner.namespace_id;
    assert(!cn_scalar_write(f.context, &f.owner, f.ref.context_epoch, 10, (cn_u64)f.memory));
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, 1, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause && !r.bytes_done);
    cleanup(&f);
    setup(&f, PARTIAL);
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, SIZE, CU_TO_DOMAIN, PAGE);
    CHECK(!r.cause && r.bytes_done == SIZE && f.faults == 1);
    CHECK(!memcmp(buffer, f.memory, SIZE)); cleanup(&f);
    setup(&f, STUCK);
    r = cu_copy(&ops, &f, &f.ref, 0, buffer, SIZE, CU_TO_DOMAIN, PAGE);
    CHECK(r.cause == CU_REFUSED && !r.bytes_done && f.faults == 4 && f.holds == 0);
    cleanup(&f);
    printf("checked-copy checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
