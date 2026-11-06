/* SPDX-License-Identifier: MIT */
#include <capstone/native-context.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Deliberately synthetic namespace/copy doubles. Payload word 1's low four
 * bits name a grant here; this is NOT a Capstone decoder or authority source. */
struct namespace {
    struct cn_owner owner;
    struct cn_translation translation;
    unsigned live[16];
};
struct fixture {
    struct namespace spaces[4];
    unsigned kind, fault;
    unsigned clone_rollbacks;
    int share_root, bad_event;
};
static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr, "FAIL line %u: %s\n", (unsigned)__LINE__, #x); exit(1); } } while (0)
static struct namespace *space(struct fixture *f, cn_u64 id)
{
    for (unsigned i = 0; i < 4; ++i) if (f->spaces[i].owner.namespace_id == id) return &f->spaces[i];
    return NULL;
}
static void namespace_init(struct namespace *n, cn_u64 id)
{
    memset(n, 0, sizeof(*n));
    n->owner = (struct cn_owner){id * 10, id * 100, id, 1};
    n->translation = (struct cn_translation){0x8000000000000000ULL | id, 0x1000 * id,
                                            1, id * 100, id, 1};
    for (unsigned i = 0; i < 16; ++i) n->live[i] = 1;
}
static int validate(void *opaque, const struct cn_owner *o, const struct cn_translation *t)
{
    struct namespace *n = space(opaque, o->namespace_id);
    return n && n->owner.namespace_epoch == o->namespace_epoch &&
           n->owner.address_space_id == o->address_space_id &&
           !memcmp(t, &n->translation, sizeof(*t)) ? 0 : -ESTALE;
}
static int clone_namespace(void *opaque, const struct cn_owner *parent, const struct cn_translation *from,
                            const struct cn_owner *child, struct cn_translation *to)
{
    struct fixture *f = opaque;
    struct namespace *p = space(f, parent->namespace_id), *n = &f->spaces[2];
    if (!p || from->root_pa != p->translation.root_pa) return -ESTALE;
    namespace_init(n, child->namespace_id); n->owner = *child;
    memcpy(n->live, p->live, sizeof(n->live));
    if (f->share_root) n->translation.root_pa = p->translation.root_pa;
    *to = n->translation;
    return 0;
}
static void abort_clone(void *opaque, const struct cn_owner *child)
{
    struct fixture *f = opaque;
    struct namespace *n = space(f, child->namespace_id);
    CHECK(n != NULL); memset(n, 0, sizeof(*n)); ++f->clone_rollbacks;
}
static int authorize(void *opaque, const struct cn_owner *o, const struct cn_register *r,
                     cn_u64 off, cn_u64 length, int write)
{
    struct namespace *n = space(opaque, o->namespace_id);
    (void)write;
    if (!n || !n->live[r->payload[1] & 15]) return -ESTALE;
    return off <= 32 && length <= 32 - off ? 0 : -EACCES;
}
static int step(void *opaque, struct cn_frame *f, struct cn_event *event)
{
    struct fixture *test = opaque;
    event->kind = test->kind; event->fault_class = test->fault;
    event->fault_pc = f->pc; event->cause = 0x123; event->address = 0x456;
    if (test->bad_event) event->bytes = 0;
    return 0;
}
static int complete_service(void *opaque, struct cn_frame *f, cn_u64 reply)
{
    (void)opaque;
    memset(&f->x[10], 0, sizeof(f->x[10])); f->x[10].payload[0] = reply;
    f->pc += 4;
    /* The synthetic PCC encoding in this test puts the cursor in word 0. */
    f->ccsr[0].payload[0] = f->pc;
    return 0;
}
static void seed(struct cn_frame *f, const struct namespace *n, unsigned salt)
{
    memset(f, 0, sizeof(*f));
    f->version = CN_ABI_VERSION; f->bytes = sizeof(*f); f->features = CN_FEATURES_V1;
    f->translation = n->translation; f->pc = 0x8000 + salt * 0x100;
    f->status = 0x4000 + salt; f->fcsr = (salt * 17) & 255;
    for (unsigned i = 1; i < CN_REGISTER_COUNT; ++i) {
        f->x[i].payload[0] = (0xfedcba9876543210ULL ^ (salt * 101 + i));
        f->x[i].payload[1] = (0x123456789abcdef0ULL ^ (salt * 131)) & ~15ULL;
        f->x[i].payload[1] |= i & 15;
        f->x[i].exact_base = i + salt * 33; f->x[i].exact_end = i + salt * 33 + 65;
        f->x[i].flags = (i & 1 ? CN_REG_TAG : CN_REG_RAW_TAG) | CN_REG_EXACT_BOUNDS;
        f->f[i] = 0x7ff0000000000001ULL ^ (salt * 177 + i); /* NaN bit patterns */
    }
    f->f[0] = 0x8000000000000000ULL; /* negative zero */
    for (unsigned i = 0; i < CN_CAP_CSR_COUNT; ++i) f->ccsr[i] = f->x[i+1];
    f->ccsr[0].payload[0] = f->pc;
}
static cn_u64 epoch(struct cn_context *c, const struct cn_owner *o)
{
    cn_u64 result = 0;
    CHECK(cn_get_epoch(c, o, &result) == 0);
    return result;
}
static void same(struct cn_context *c, const struct cn_owner *o, const struct cn_frame *want)
{
    struct cn_frame got;
    CHECK(cn_snapshot(c, o, &got) == 0);
    CHECK(!memcmp(&got, want, sizeof(got)));
}
static void run(struct cn_context *c, const struct cn_owner *o, struct cn_event *event)
{
    CHECK(cn_run(c, o, epoch(c, o), CN_RESUME_RETRY, 0, event) == 0);
}
int main(void)
{
    struct fixture f = {.kind = CN_EVENT_QUANTUM};
    namespace_init(&f.spaces[0], 1); namespace_init(&f.spaces[1], 2);
    struct cn_owner a = f.spaces[0].owner, b = f.spaces[1].owner;
    struct cn_ops ops = {.validate_namespace = validate, .clone_namespace = clone_namespace,
        .abort_clone = abort_clone, .authorize = authorize, .step = step, .complete_service = complete_service};
    struct cn_engine *e = NULL;
    CHECK(cn_engine_create(CN_BACKEND_ARCH_RETURN, &ops, &f, &e) == -EOPNOTSUPP && !e);
    CHECK(cn_engine_create(CN_BACKEND_IOCTL, &ops, &f, &e) == 0);
    struct cn_frame fa, fb, bad, h1, h2;
    seed(&fa, &f.spaces[0], 3); seed(&fb, &f.spaces[1], 9);
    seed(&h1, &f.spaces[0], 21); seed(&h2, &f.spaces[0], 37);
    struct cn_context *ca, *cb, *rejected = NULL;
    bad = fa; ++bad.version;
    CHECK(cn_create(e, &a, &bad, &rejected) == -EPROTONOSUPPORT && !rejected);
    bad = fa; bad.features |= 1ULL << 63;
    CHECK(cn_create(e, &a, &bad, &rejected) == -EPROTONOSUPPORT && !rejected);
    bad = fa; bad.x[0].flags = CN_REG_TAG;
    CHECK(cn_create(e, &a, &bad, &rejected) == -EINVAL && !rejected);
    CHECK(cn_create(e, &a, &fa, &ca) == 0);
    CHECK(cn_create(e, &b, &fb, &cb) == 0);
    struct cn_event event;
    struct cn_capref ref, wrong;
    CHECK(cn_get_capref(ca, &a, 7, &ref) == 0);
    CHECK(cn_authorize(ca, &a, &ref, 0, 32, 0) == 0);
    CHECK(cn_authorize(ca, &b, &ref, 0, 1, 0) == -EPERM);
    wrong = ref; ++wrong.context_epoch;
    CHECK(cn_authorize(ca, &a, &wrong, 0, 1, 0) == -ESTALE);
    CHECK(cn_authorize(cb, &b, &ref, 0, 1, 0) == -ESTALE);
    CHECK(cn_authorize(ca, &a, &ref, UINT64_MAX, 2, 0) == -EOVERFLOW);
    CHECK(cn_authorize(ca, &a, &ref, 32, 1, 0) == -EACCES);
    CHECK(cn_authorize(ca, &a, &ref, 32, 0, 0) == 0);
    CHECK(cn_authorize(ca, &b, &ref, 0, 0, 0) == -EPERM);
    CHECK(cn_run(ca, &b, 1, CN_RESUME_RETRY, 0, &event) == -EPERM);
    CHECK(cn_run(ca, &a, 99, CN_RESUME_RETRY, 0, &event) == -ESTALE);
    for (unsigned round = 0; round < 128; ++round) {
        run(ca, &a, &event); same(ca, &a, &fa);
        CHECK(event.restart_pc == fa.pc && event.fault_pc == fa.pc);
        run(cb, &b, &event); same(cb, &b, &fb);
        same(ca, &a, &fa);
    }
    CHECK(cn_authorize(ca, &a, &ref, 0, 1, 0) == -ESTALE);
    struct cn_signal_token t1, t2, replay;
    CHECK(cn_signal_push(ca, &a, epoch(ca, &a), &h1, &t1) == 0);
    run(cb, &b, &event); same(cb, &b, &fb);
    run(ca, &a, &event); same(ca, &a, &h1);
    CHECK(cn_signal_push(ca, &a, epoch(ca, &a), &h2, &t2) == 0);
    CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &t1) == -EPERM);
    run(cb, &b, &event); run(ca, &a, &event); same(ca, &a, &h2);
    CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &t2) == 0); same(ca, &a, &h1);
    CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &t1) == 0); same(ca, &a, &fa);
    CHECK(cn_signal_push(ca, &a, epoch(ca, &a), &h1, &replay) == 0);
    CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &t1) == -EPERM);
    CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &replay) == 0);
    struct cn_signal_token stack[CN_MAX_SIGNAL_DEPTH];
    for (unsigned i = 0; i < CN_MAX_SIGNAL_DEPTH; ++i)
        CHECK(cn_signal_push(ca, &a, epoch(ca, &a), &h1, &stack[i]) == 0);
    CHECK(cn_signal_push(ca, &a, epoch(ca, &a), &h2, &replay) == -EOVERFLOW);
    for (unsigned i = CN_MAX_SIGNAL_DEPTH; i-- > 0;)
        CHECK(cn_signal_return(ca, &a, epoch(ca, &a), &stack[i]) == 0);
    same(ca, &a, &fa);
    struct cn_owner child_owner = {30, 300, 3, 1};
    struct cn_context *child = NULL;
    CHECK(cn_clone(ca, &a, &b, &child) == -EEXIST && !child);
    same(cb, &b, &fb);
    f.share_root = 1;
    CHECK(cn_clone(ca, &a, &child_owner, &child) == -EPERM && !child);
    CHECK(f.clone_rollbacks == 1);
    f.share_root = 0;
    CHECK(cn_clone(ca, &a, &child_owner, &child) == 0);
    struct cn_frame fc = fa; fc.translation = f.spaces[2].translation;
    CHECK(cn_get_capref(child, &child_owner, 7, &ref) == 0);
    CHECK(cn_authorize(child, &child_owner, &ref, 0, 1, 0) == 0);
    f.spaces[2].live[7] = 0;
    CHECK(cn_authorize(child, &child_owner, &ref, 0, 1, 0) == -ESTALE);
    CHECK(cn_get_capref(ca, &a, 7, &ref) == 0);
    CHECK(cn_authorize(ca, &a, &ref, 0, 1, 0) == 0);
    for (unsigned i = 0; i < 64; ++i) {
        run(child, &child_owner, &event); same(child, &child_owner, &fc);
        run(ca, &a, &event); same(ca, &a, &fa);
    }
    CHECK(cn_scalar_write(ca, &a, epoch(ca, &a), 7, 42) == 0);
    memset(&fa.x[7], 0, sizeof(fa.x[7])); fa.x[7].payload[0] = 42;
    same(ca, &a, &fa);
    CHECK(cn_get_capref(ca, &a, 7, &ref) == 0);
    CHECK(cn_authorize(ca, &a, &ref, 0, 1, 0) == -EACCES);
    f.kind = CN_EVENT_PAGE_FAULT; f.fault = CN_FAULT_TRANSLATION;
    run(ca, &a, &event);
    CHECK(event.restart_pc == fa.pc && event.fault_class == CN_FAULT_TRANSLATION);
    f.kind = CN_EVENT_SERVICE; f.fault = CN_FAULT_NONE;
    run(ca, &a, &event);
    CHECK(cn_run(ca, &a, epoch(ca, &a), CN_RESUME_RETRY, 0, &event) == -EINVAL);
    cn_u64 old_epoch = epoch(ca, &a), old_sequence = event.sequence;
    f.kind = CN_EVENT_QUANTUM;
    CHECK(cn_run(ca, &a, old_epoch, CN_RESUME_REPLY, 123, &event) == 0);
    CHECK(event.sequence > old_sequence && event.restart_pc == fa.pc + 4);
    complete_service(&f, &fa, 123); same(ca, &a, &fa);
    CHECK(cn_run(ca, &a, old_epoch, CN_RESUME_REPLY, 123, &event) == -ESTALE);
    CHECK(cn_get_capref(child, &child_owner, 5, &wrong) == 0);
    CHECK(cn_destroy(child, &a) == -EPERM);
    CHECK(cn_destroy(child, &child_owner) == 0);
    CHECK(cn_authorize(child, &child_owner, &wrong, 0, 1, 0) == -ESTALE);
    CHECK(cn_create(e, &child_owner, &fc, &child) == 0);
    CHECK(cn_authorize(child, &child_owner, &wrong, 0, 1, 0) == -ESTALE);
    ++f.spaces[2].translation.translation_epoch;
    CHECK(cn_run(child, &child_owner, epoch(child, &child_owner), CN_RESUME_RETRY, 0, &event) == -ESTALE);
    --f.spaces[2].translation.translation_epoch;
    f.bad_event = 1;
    CHECK(cn_run(child, &child_owner, epoch(child, &child_owner), CN_RESUME_RETRY, 0, &event) == -EPROTO);
    CHECK(cn_get_epoch(child, &child_owner, &old_epoch) == -ESTALE);
    f.bad_event = 0; f.kind = CN_EVENT_CAP_FAULT; f.fault = CN_FAULT_LIFETIME;
    run(ca, &a, &event);
    CHECK(event.fault_class == CN_FAULT_LIFETIME);
    CHECK(cn_get_epoch(ca, &a, &old_epoch) == -ESTALE);
    struct cn_counters counters;
    cn_get_counters(e, &counters);
    CHECK(counters.evictions > 300 && counters.restorations > 300);
    printf("PASS context reference: %u checks; entries=%llu evictions=%llu restorations=%llu bytes_moved=%llu\n",
           checks, (unsigned long long)counters.entries, (unsigned long long)counters.evictions,
           (unsigned long long)counters.restorations, (unsigned long long)counters.bytes_moved);
    cn_engine_free(e);
    return 0;
}
