/* SPDX-License-Identifier: MIT */
/* Shared dispatcher. All memory authority and Linux issue live in providers. */
#include <capstone/native-syscall.h>
#ifdef __KERNEL__
#include <linux/errno.h>
#include <linux/slab.h>
#include <linux/string.h>
#define calloc(n, size) kcalloc(n, size, GFP_KERNEL)
#define free(p) kfree(p)
#ifndef UINT64_MAX
#define UINT64_MAX (~(cn_u64)0)
#define INT64_MAX ((cn_u64)(UINT64_MAX >> 1))
#endif
#else
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#endif

struct ns_session {
    struct cn_context *context;
    struct cn_owner owner;
    struct ns_ops ops;
    void *opaque;
    cn_u64 last_epoch;
    struct ns_counters counters;
};
struct snapshot {
    struct ns_invocation call;
    struct cn_capref refs[6];
    cn_u64 offsets[6];
    struct ns_iov iov[NS_MAX_IOV];
    unsigned tagged, nulls, imported;
    size_t allocated;
};
static int reads(unsigned kind)
{
    return kind == CAPSTONE_ARG_IN || kind == CAPSTONE_ARG_OPT_IN ||
        kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
}
static int writes(unsigned kind)
{
    return kind == CAPSTONE_ARG_OUT || kind == CAPSTONE_ARG_OPT_OUT ||
        kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
}
static int optional(unsigned kind)
{
    return kind >= CAPSTONE_ARG_OPT_IN && kind <= CAPSTONE_ARG_OPT_STR;
}
static int allocate(struct snapshot *s, struct ns_buffer *b, size_t length)
{
    if (length > NS_MAX_BYTES - s->allocated) return -E2BIG;
    b->data = calloc(1, length ? length : 1);
    if (!b->data) return -ENOMEM;
    b->length = length;
    s->allocated += length;
    return 0;
}
static int copy(struct ns_session *s, const struct cn_capref *ref, cn_u64 offset,
                void *buffer, size_t length, enum ns_direction direction,
                struct ns_result *result, size_t *done)
{
    *done = 0;
    if (!s->ops.copy) return -ENOSYS;
    struct ns_copy_result r = s->ops.copy(s->opaque, s->context, &s->owner,
                                         ref, offset, buffer, length, direction);
    ++s->counters.copy_calls;
    if (r.bytes_done > length || (!r.cause && r.bytes_done != length)) return -EPROTO;
    *done = r.bytes_done;
    if (direction == NS_FROM_DOMAIN) s->counters.bytes_in += r.bytes_done;
    else { s->counters.bytes_out += r.bytes_done; result->bytes_out += r.bytes_done; }
    if (r.cause) { result->cause = r.cause; return -EFAULT; }
    return 0;
}
static int authorize(struct ns_session *s, const struct cn_capref *ref, cn_u64 offset, size_t length, int write)
{
    /* Preflight bounds before issue; never a permit to copy after blocking. */
    return cn_authorize(s->context, &s->owner, ref, offset, length, write);
}
int ns_session_admit(struct ns_session *s, cn_u64 epoch)
{
    if (!s || !epoch || epoch <= s->last_epoch) return -ESTALE;
    s->last_epoch = epoch; /* Includes errors: no replay of a side effect. */
    return 0;
}
static int capture(struct ns_session *s, cn_u64 epoch, struct snapshot *p,
                   const struct ns_request *request)
{
    struct cn_frame *f;
    cn_u64 current_epoch;
    int rc = cn_get_epoch(s->context, &s->owner, &current_epoch);
    if (rc) return rc;
    if (current_epoch != epoch) return -ESTALE;
    if ((rc = ns_session_admit(s, epoch))) return rc;
    if (request) {
        p->call.descriptor = ns_descriptor(request->nr);
        if (!p->call.descriptor) return -ENOSYS;
        memcpy(p->call.scalar, request->scalar, sizeof(request->scalar));
        memcpy(p->refs, request->refs, sizeof(request->refs));
#ifndef NS_MUTATE_OFFSET
        memcpy(p->offsets, request->offsets, sizeof(request->offsets));
#endif
        p->tagged = request->tagged;
        p->nulls = request->nulls;
        /* Nested capability import with a shifted outer pointer needs its own
         * provider. The initial exchange adapter refuses vector calls. */
        if (p->call.descriptor->route == NS_VECTOR) return -ENOSYS;
        return 0;
    }
    f = calloc(1, sizeof(*f));
    if (!f) return -ENOMEM;
    rc = cn_snapshot(s->context, &s->owner, f);
    if (rc) goto out;
    if (f->x[17].flags || f->x[16].flags || f->x[17].payload[1] || f->x[16].payload[1] ||
        f->x[17].payload[0] != NS_SERVICE_NATIVE) { rc = -EPROTO; goto out; }
    p->call.descriptor = ns_descriptor(f->x[16].payload[0]);
    if (!p->call.descriptor) { rc = -ENOSYS; goto out; }
    for (unsigned i = 0; i < 6; ++i) {
        const struct cn_register *r = &f->x[10 + i];
        if (r->flags & CN_REG_RAW_TAG) { rc = -EFAULT; goto out; }
        if (r->flags & CN_REG_TAG) p->tagged |= 1U << i;
        if (!r->flags && !r->payload[0] && !r->payload[1]) p->nulls |= 1U << i;
        if (!(r->flags & CN_REG_TAG) && r->payload[1]) { rc = -EINVAL; goto out; }
        p->call.scalar[i] = r->payload[0];
        if ((rc = cn_get_capref(s->context, &s->owner, 10 + i, &p->refs[i]))) goto out;
    }
out:
    free(f);
    return rc;
}
static int vector_snapshot(struct ns_session *s, struct snapshot *p, struct ns_result *result)
{
    cn_u64 count = p->call.scalar[2];
    int input = p->call.descriptor->nr == CAPSTONE_SYS_writev ||
                p->call.descriptor->nr == CAPSTONE_SYS_pwritev;
    size_t total = 0, done;
    int rc;
    if (count > NS_MAX_IOV) return -EINVAL;
    /* Even a zero-length nonnull vector must be an authenticated reference. */
    if (!count && (p->nulls & 2)) { p->call.scalar[1] = 0; return 0; }
    if (!(p->tagged & 2)) return -EFAULT;
    if (!s->ops.copy || !s->ops.snapshot_iov || !s->ops.release_iov) return -ENOSYS;
    rc = s->ops.snapshot_iov(s->opaque, s->context, &s->owner, &p->refs[1], count, p->iov);
    if (rc) return rc; /* Provider rolls back on failure. */
    p->imported = 1;
    p->call.vector_count = count;
    ++s->counters.iov_snapshots;
    p->call.scalar[1] = 0;
    for (unsigned i = 0; i < count; ++i) {
        cn_u64 length = p->iov[i].length;
        if (length > INT64_MAX - total) return -EINVAL;
        total += length;
        if (length > NS_MAX_BYTES) return -E2BIG;
        if ((rc = authorize(s, &p->iov[i].base, 0, length, !input))) return rc;
        if ((rc = allocate(p, &p->call.vector[i], length))) return rc;
        if (input && (rc = copy(s, &p->iov[i].base, 0, p->call.vector[i].data,
                               length, NS_FROM_DOMAIN, result, &done))) return rc;
    }
    return 0;
}
static int prepare(struct ns_session *s, struct snapshot *p, struct ns_result *result)
{
    const struct ns_descriptor *d = p->call.descriptor;
    const struct capstone_delegate_shape *shape = capstone_delegate_shape(d->nr);
    int rc;
    size_t done;
    if (d->route == NS_REFUSE || d->route == NS_GRANT || d->nr == CAPSTONE_SYS_pipe2)
        return -ENOSYS;
    if (!shape) return -ENOSYS;
    if (d->nr == CAPSTONE_NR_FCNTL_LOCK && p->call.scalar[1] != NS_F_GETLK &&
        p->call.scalar[1] != NS_F_SETLK && p->call.scalar[1] != NS_F_SETLKW)
        return -EINVAL;
    /* No tagged values silently treated as scalars, including ignored args. */
    for (unsigned i = 0; i < 6; ++i) {
        if ((i >= shape->argc || shape->args[i].kind == CAPSTONE_ARG_INT) &&
            (p->tagged & (1U << i))) return -EINVAL;
        if (i >= shape->argc) p->call.scalar[i] = 0;
    }
    if (d->route == NS_VECTOR) return vector_snapshot(s, p, result);
    /* Length sources stay in this immutable scalar snapshot. */
    cn_u64 scalars[6];
    memcpy(scalars, p->call.scalar, sizeof(scalars));
    for (unsigned i = 0; i < shape->argc; ++i) {
        struct capstone_delegate_arg effective = shape->args[i];
        /* SETLK/W only consume flock. Never require write authority or publish
         * a buffer after a successful side effect for these commands. */
        if (d->nr == CAPSTONE_SYS_clock_nanosleep && i == 3 && (p->call.scalar[1] & 1)) {
            p->call.scalar[i] = 0; /* Absolute sleeps ignore the remaining pointer. */
            continue;
        }
        if (d->nr == CAPSTONE_NR_FCNTL_LOCK && i == 2 && scalars[1] != NS_F_GETLK)
            effective.kind = CAPSTONE_ARG_IN;
        const struct capstone_delegate_arg *a = &effective;
        cn_u64 length = 0;
        if (a->kind == CAPSTONE_ARG_INT) continue;
        if (p->offsets[i] > UINT64_MAX - NS_MAX_BYTES) return -EOVERFLOW;
        p->call.scalar[i] = 0;
        if (optional(a->kind) && (p->nulls & (1U << i))) continue;
        if (!s->ops.copy) return -ENOSYS;
        if (a->kind == CAPSTONE_ARG_STR || a->kind == CAPSTONE_ARG_OPT_STR) {
            if (!(p->tagged & (1U << i))) return -EFAULT;
            if ((rc = allocate(p, &p->call.buffer[i], NS_MAX_PATH))) return rc;
            char *path = p->call.buffer[i].data;
            size_t j;
            /* Byte copies do not read past the first NUL or a tight bound. */
            for (j = 0; j < NS_MAX_PATH; ++j) {
                rc = copy(s, &p->refs[i], p->offsets[i] + j, path + j, 1, NS_FROM_DOMAIN, result, &done);
                if (rc) return rc;
                if (!path[j]) break;
            }
            if (j == NS_MAX_PATH) return -ENAMETOOLONG;
            p->call.buffer[i].length = j + 1;
            continue;
        }
        switch (a->length) {
        case CAPSTONE_LEN_FIXED: length = a->size; break;
        case CAPSTONE_LEN_ARG: length = scalars[a->size]; break;
        case CAPSTONE_LEN_ARG_SCALED:
            if (!a->scale || scalars[a->size] > UINT64_MAX / a->scale) return -EOVERFLOW;
            length = scalars[a->size] * a->scale; break;
        default: return -ENOSYS; /* Nested word/pointer layouts need a converter. */
        }
        if (length > NS_MAX_BYTES) return -E2BIG;
        if (!(p->tagged & (1U << i))) {
            /* Linux-compatible NULL with a zero dynamic length; never accept
             * a forged nonzero scalar, even if the requested length is zero. */
            if (!length && (p->nulls & (1U << i)) && a->length != CAPSTONE_LEN_FIXED)
                continue;
            return -EFAULT;
        }
        if (reads(a->kind) && (rc = authorize(s, &p->refs[i], p->offsets[i], length, 0))) return rc;
        if (writes(a->kind) && (rc = authorize(s, &p->refs[i], p->offsets[i], length, 1))) return rc;
        if ((rc = allocate(p, &p->call.buffer[i], length))) return rc;
        if (reads(a->kind) && (rc = copy(s, &p->refs[i], p->offsets[i], p->call.buffer[i].data,
                                       length, NS_FROM_DOMAIN, result, &done))) return rc;
        /* A zero-length tagged output still passes through U's interface. */
        if (!length && !reads(a->kind) && (rc = copy(s, &p->refs[i], p->offsets[i],
            p->call.buffer[i].data, 0, NS_TO_DOMAIN, result, &done))) return rc;
    }
#ifdef NS_MUTATE_OVERLAP
    if (d->nr == CAPSTONE_SYS_newfstatat)
        memset(p->call.buffer[1].data, 0, p->call.buffer[1].length);
#endif
    return 0;
}
static int byte_result(cn_u64 nr)
{
    return nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64 ||
        nr == CAPSTONE_SYS_getrandom || nr == CAPSTONE_SYS_getdents64 || nr == CAPSTONE_SYS_getcwd ||
        nr == CAPSTONE_SYS_readv || nr == CAPSTONE_SYS_preadv;
}
static int partial_result(cn_u64 nr)
{
    return nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64 ||
        nr == CAPSTONE_SYS_readv || nr == CAPSTONE_SYS_preadv || nr == CAPSTONE_SYS_getrandom;
}
static int output(struct ns_session *s, struct snapshot *p, struct ns_result *result)
{
    const struct ns_descriptor *d = p->call.descriptor;
    size_t done;
    int rc;
    if (result->value < 0 && result->value != -EINTR) return 0;
    if (result->value < 0 && d->nr != CAPSTONE_SYS_nanosleep &&
        d->nr != CAPSTONE_SYS_clock_nanosleep && d->nr != CAPSTONE_SYS_pselect6) return 0;
    if (d->route == NS_VECTOR) {
#ifdef NS_MUTATE_IOV
        if (s->ops.snapshot_iov(s->opaque, s->context, &s->owner, &p->refs[1],
                               p->call.vector_count, p->iov)) return -EFAULT;
#endif
        size_t total = 0;
        for (unsigned i = 0; i < p->call.vector_count; ++i) total += p->iov[i].length;
        if ((cn_u64)result->value > total) return -EPROTO;
        if (!byte_result(d->nr)) return 0;
        size_t left = result->value;
        for (unsigned i = 0; i < p->call.vector_count && left; ++i) {
            size_t n = p->iov[i].length < left ? p->iov[i].length : left;
            rc = copy(s, &p->iov[i].base, 0, p->call.vector[i].data, n,
                      NS_TO_DOMAIN, result, &done);
            if (rc) return rc;
            left -= n;
        }
        return 0;
    }
    const struct capstone_delegate_shape *shape = capstone_delegate_shape(d->nr);
    if ((d->nr == CAPSTONE_SYS_write || d->nr == CAPSTONE_SYS_pwrite64) &&
        (cn_u64)result->value > p->call.buffer[1].length) return -EPROTO;
    /* Reject an invalid backend count before any publication. */
    if (byte_result(d->nr)) {
        size_t total = 0;
        for (unsigned i = 0; i < shape->argc; ++i)
            if (writes(shape->args[i].kind)) total += p->call.buffer[i].length;
        if ((cn_u64)result->value > total) return -EPROTO;
    }
    for (unsigned i = 0; i < shape->argc; ++i) {
        if (d->nr == CAPSTONE_SYS_clock_nanosleep && i == 3 && (p->call.scalar[1] & 1)) continue;
        if (d->nr == CAPSTONE_NR_FCNTL_LOCK && p->call.scalar[1] != NS_F_GETLK) continue;
        if (!writes(shape->args[i].kind) || !p->call.buffer[i].data) continue;
        size_t n = capstone_delegate_result_bytes(d->nr, i, p->call.buffer[i].length, result->value);
        if (n && (rc = copy(s, &p->refs[i], p->offsets[i], p->call.buffer[i].data, n,
                           NS_TO_DOMAIN, result, &done))) return rc;
    }
    return 0;
}
int ns_session_create(struct cn_context *context, const struct cn_owner *owner,
                      const struct ns_ops *ops, void *opaque, struct ns_session **out)
{
    cn_u64 epoch;
    if (!out) return -EINVAL;
    *out = NULL;
    if (!context || !owner || !ops || !ops->acquire || !ops->issue || !ops->release)
        return -EINVAL;
    int rc = cn_get_epoch(context, owner, &epoch);
    if (rc) return rc;
    struct ns_session *s = calloc(1, sizeof(*s));
    if (!s) return -ENOMEM;
    s->context = context; s->owner = *owner; s->ops = *ops; s->opaque = opaque;
    *out = s;
    return 0;
}
void ns_session_free(struct ns_session *s) { free(s); }
void ns_get_counters(const struct ns_session *s, struct ns_counters *out) { *out = s->counters; }
int ns_dispatch_request(struct ns_session *s, cn_u64 epoch,
                        const struct ns_request *request, struct ns_result *result)
{
    if (!s || !result) return -EINVAL;
    *result = (struct ns_result){0};
    struct snapshot *p = calloc(1, sizeof(*p));
    if (!p) { result->value = -ENOMEM; return 0; }
    void *held = NULL;
    unsigned acquired = 0;
    int rc = capture(s, epoch, p, request);
    if (!rc) rc = prepare(s, p, result);
    if (!rc) {
        rc = s->ops.acquire(s->opaque, &p->call, &held);
        acquired = !rc;
    }
    if (!rc) {
        struct ns_completion completion = s->ops.issue(s->opaque, &p->call, held);
        ++s->counters.issued;
        result->value = completion.value;
        result->effects = completion.effects;
        if (completion.restart) {
#ifndef NS_MUTATE_RESTART
            if (completion.value != -EINTR || completion.effects) rc = -EPROTO;
            else
#endif
                result->restart = 1;
        } else rc = output(s, p, result);
    }
    if (rc) {
        result->restart = 0;
        result->value = result->bytes_out && p->call.descriptor &&
            partial_result(p->call.descriptor->nr) ? (int64_t)result->bytes_out : rc;
    }
    if (acquired) s->ops.release(s->opaque, held);
    if (p->imported) s->ops.release_iov(s->opaque, s->context, p->iov, p->call.vector_count);
    for (unsigned i = 0; i < 6; ++i) free(p->call.buffer[i].data);
    for (unsigned i = 0; i < p->call.vector_count; ++i) free(p->call.vector[i].data);
    free(p);
    return 0;
}

int ns_dispatch(struct ns_session *s, cn_u64 epoch, struct ns_result *result)
{ return ns_dispatch_request(s, epoch, NULL, result); }
#ifdef __KERNEL__
#undef calloc
#undef free
#endif
