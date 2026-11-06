/* SPDX-License-Identifier: MIT */
/* Synthetic authority/resource fixtures; no hardware or Linux-VFS claim. */
#include <capstone/native-syscall.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(label, expression) do { ++checks; if (!(expression)) { \
    ++failures; printf("FAIL %s line %u\n", label, __LINE__); } } while (0)
struct object { unsigned char data[512], tags[32]; unsigned live, generation, rights; cn_u64 ns; };
struct file_object { unsigned identity, safe, permissions; };
struct fixture {
    struct object objects[8];
    struct file_object normal, private_memory, *fd[8], *grant[8], *held;
    struct cn_owner owner;
    struct cn_engine *engine;
    struct cn_context *context;
    struct ns_session *session;
    struct cn_frame next;
    struct ns_iov vectors[NS_MAX_IOV];
    cn_u64 epoch;
    unsigned issue_count, held_count, released, iov_released, copies;
    unsigned revoke_issue, recycle_mid_copy, stale_retry, fail_after, bad_copy;
    unsigned replace_fd, inherited_filter, shared_fs, namespace_bad, import_bad;
    unsigned mutate_vectors, force_completion;
    struct ns_completion completion;
    unsigned char written[512];
    size_t written_bytes;
};
static int valid_namespace(void *opaque, const struct cn_owner *o, const struct cn_translation *t)
{
    struct fixture *f = opaque;
    return f->namespace_bad || o->namespace_id != t->namespace_id ? -ESTALE : 0;
}
static int auth(void *opaque, const struct cn_owner *owner, const struct cn_register *reg,
                cn_u64 offset, cn_u64 length, int write)
{
    struct fixture *f = opaque;
    cn_u64 id = reg->payload[0];
    if (id >= 8) return -EFAULT;
    const struct object *o = &f->objects[id];
    if (o->ns != owner->namespace_id || !o->live || o->generation != reg->payload[1]) return -ESTALE;
    if (!(o->rights & (write ? 2U : 1U))) return -EACCES;
    return offset > sizeof(o->data) || length > sizeof(o->data) - offset ? -EFAULT : 0;
}
static int step(void *opaque, struct cn_frame *frame, struct cn_event *event)
{
    struct fixture *f = opaque;
    *frame = f->next;
    event->kind = CN_EVENT_SERVICE;
    return 0;
}
static int complete(void *opaque, struct cn_frame *frame, cn_u64 reply)
{
    (void)opaque;
    memset(&frame->x[10], 0, sizeof(frame->x[10]));
    frame->x[10].payload[0] = reply;
    frame->pc += 4;
    return 0;
}
static struct ns_copy_result checked_copy(void *opaque, struct cn_context *context,
    const struct cn_owner *owner, const struct cn_capref *ref, cn_u64 offset,
    void *buffer, size_t length, enum ns_direction direction)
{
    struct fixture *f = opaque;
    struct ns_copy_result result = {0};
    struct cn_frame frame;
    ++f->copies;
    if (f->bad_copy) return (struct ns_copy_result){length + 1, 0};
    if (cn_snapshot(context, owner, &frame)) return (struct ns_copy_result){0, 24};
    if (ref->frame_slot >= CN_REGISTER_COUNT) return (struct ns_copy_result){0, 24};
    cn_u64 id = frame.x[ref->frame_slot].payload[0];
    if (id >= 8) return (struct ns_copy_result){0, 24};
    do {
        size_t n = length - result.bytes_done;
        if (n > 16) n = 16;
        /* This single-threaded model's loop represents an excluded chunk.
         * Revocation is injected between chunks, never inside one. */
#ifndef NS_MUTATE_RECHECK
        if (cn_authorize(context, owner, ref, offset + result.bytes_done, n,
                         direction == NS_TO_DOMAIN)) { result.cause = 25; break; }
#endif
        if (f->fail_after && result.bytes_done >= f->fail_after) { result.cause = 29; break; }
        if (offset + result.bytes_done > 512 || n > 512 - offset - result.bytes_done) {
            result.cause = 29; break;
        }
        if (direction == NS_FROM_DOMAIN)
            memcpy((char *)buffer + result.bytes_done, f->objects[id].data + offset + result.bytes_done, n);
        else {
            memcpy(f->objects[id].data + offset + result.bytes_done, (char *)buffer + result.bytes_done, n);
#ifndef NS_MUTATE_TAGS
            for (size_t j = 0; j < n; ++j) f->objects[id].tags[(offset + result.bytes_done + j) / 16] = 0;
#endif
        }
        result.bytes_done += n;
        if (n && f->recycle_mid_copy) {
            f->recycle_mid_copy = 0;
            ++f->objects[id].generation;
            memset(f->objects[id].data + result.bytes_done, 0x5a, 512 - result.bytes_done);
        }
    } while (result.bytes_done < length);
    return result;
}
static int snapshot_iov(void *opaque, struct cn_context *context, const struct cn_owner *owner,
    const struct cn_capref *ref, size_t count, struct ns_iov *out)
{
    struct fixture *f = opaque;
    /* The fixture's protected vectors supply authenticated references. Payload
     * bytes in objects[0] are deliberately ignored, including forged caprefs. */
    if (f->import_bad || cn_authorize(context, owner, ref, 0, count * 32, 0)) return -EFAULT;
    memcpy(out, f->vectors, count * sizeof(*out));
    return 0;
}
static void release_iov(void *opaque, struct cn_context *context, struct ns_iov *iov, size_t count)
{
    struct fixture *f = opaque;
    (void)context; (void)iov; (void)count;
    ++f->iov_released;
}
static int acquire(void *opaque, const struct ns_invocation *call, void **held)
{
    struct fixture *f = opaque;
    cn_u64 nr = call->descriptor->nr;
    if (f->shared_fs || f->inherited_filter) return -EPERM;
    if (nr == CAPSTONE_SYS_ioctl) {
#ifndef NS_MUTATE_HIDDEN
        if (call->scalar[1] != 0x5450 && call->scalar[1] != 0x5451) return -ENOSYS;
#endif
    }
    if (nr == CAPSTONE_SYS_fcntl && call->scalar[1] != 1 && call->scalar[1] != 2 && call->scalar[1] != 3)
        return -ENOSYS;
    if (nr == CAPSTONE_NR_FCNTL_LOCK && call->scalar[1] != 5 && call->scalar[1] != 6 && call->scalar[1] != 7)
        return -ENOSYS;
    if (nr == CAPSTONE_SYS_prlimit64 && (call->scalar[0] || call->buffer[2].data)) return -EPERM;
    if (nr == CAPSTONE_SYS_rt_sigprocmask && call->scalar[3] != 8) return -EINVAL;
    struct file_object *pin = NULL;
    if (nr == CAPSTONE_SYS_openat || nr == CAPSTONE_SYS_newfstatat || nr == CAPSTONE_SYS_statx || nr == CAPSTONE_SYS_unlinkat) {
        const char *path = call->buffer[1].data;
        /* Fixed resolution fixture: these paths resolve to pinned objects,
         * not to a re-opened original string. No filesystem is emulated. */
        if (!strcmp(path, "/tmp/item") || !strcmp(path, "/usr/item") ||
            !strcmp(path, "/dev/null") || !strcmp(path, "/dev/zero") || !strcmp(path, "/dev/urandom"))
            pin = &f->normal;
        else pin = &f->private_memory;
#ifndef NS_MUTATE_PROC
        if (!pin->safe) return -EACCES;
#endif
        if (!(pin->permissions & 1)) return -EACCES;
    } else {
        for (unsigned i = 0; i < 6; ++i) {
            if (!(call->descriptor->fd_mask & (1U << i))) continue;
            cn_u64 fd = call->scalar[i];
            if (fd >= 8 || !f->fd[fd]) return -EBADF;
#ifndef NS_MUTATE_INHERITED
            if (!f->grant[fd]) return -EBADF;
#endif
#ifndef NS_MUTATE_REPLACEMENT
            if (f->grant[fd] && f->grant[fd] != f->fd[fd]) return -EBADF;
#endif
            pin = f->fd[fd];
        }
    }
    f->held = pin;
    ++f->held_count;
    *held = pin;
    return 0;
}
static struct ns_completion issue(void *opaque, const struct ns_invocation *call, void *held)
{
    struct fixture *f = opaque;
    cn_u64 nr = call->descriptor->nr;
    ++f->issue_count;
    if (f->replace_fd) f->fd[3] = &f->private_memory;
    if (f->revoke_issue) ++f->objects[1].generation;
    if (f->mutate_vectors) f->vectors[0].base = f->vectors[1].base;
    if (f->force_completion) return f->completion;
    if (nr == CAPSTONE_SYS_getpid) return (struct ns_completion){123, 0, 0};
    if (nr == CAPSTONE_SYS_openat) {
        f->fd[5] = f->grant[5] = held;
        return (struct ns_completion){5, 1, 0};
    }
    if (nr == CAPSTONE_SYS_close) { f->fd[call->scalar[0]] = f->grant[call->scalar[0]] = NULL; return (struct ns_completion){0, 1, 0}; }
    if (nr == CAPSTONE_SYS_newfstatat) { memset(call->buffer[2].data, 0x33, 128); return (struct ns_completion){0, 0, 0}; }
    if (nr == CAPSTONE_SYS_clock_gettime) { cn_u64 ts[2] = {7, 9}; memcpy(call->buffer[1].data, ts, sizeof(ts)); return (struct ns_completion){0, 0, 0}; }
    if (nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64) {
        size_t n = call->buffer[1].length;
        if (n > 64) n = 64;
        if (n) memset(call->buffer[1].data, held == &f->private_memory ? 0xcc : 0x42, n);
        return (struct ns_completion){n, !!n, 0};
    }
    if (nr == CAPSTONE_SYS_write || nr == CAPSTONE_SYS_pwrite64) {
        f->written_bytes = call->buffer[1].length;
        if (f->written_bytes) memcpy(f->written, call->buffer[1].data, f->written_bytes);
        return (struct ns_completion){f->written_bytes, !!f->written_bytes, 0};
    }
    if (nr == CAPSTONE_SYS_readv || nr == CAPSTONE_SYS_preadv || nr == CAPSTONE_SYS_writev || nr == CAPSTONE_SYS_pwritev) {
        size_t n = 0;
        for (unsigned i = 0; i < call->vector_count; ++i) {
            if (nr == CAPSTONE_SYS_readv || nr == CAPSTONE_SYS_preadv) memset(call->vector[i].data, 0x42 + i, call->vector[i].length);
            n += call->vector[i].length;
        }
        return (struct ns_completion){n, !!n, 0};
    }
    if (nr == CAPSTONE_SYS_ioctl) return (struct ns_completion){0, 1, 0};
    return (struct ns_completion){-ENOSYS, 0, 0};
}
static void release(void *opaque, void *held)
{ struct fixture *f = opaque; CHECK("held identity retained", held == f->held); ++f->released; }
static struct ns_ops ops = {checked_copy, snapshot_iov, release_iov, acquire, issue, release};
static struct fixture *create(void)
{
    struct fixture *f = calloc(1, sizeof(*f));
    if (!f) exit(2);
    f->owner = (struct cn_owner){1, 1, 1, 1};
    f->normal = (struct file_object){1, 1, 3};
    f->private_memory = (struct file_object){2, 0, 3};
    f->fd[3] = f->grant[3] = &f->normal;
    f->fd[4] = &f->private_memory;
    for (unsigned i = 0; i < 8; ++i) {
        f->objects[i].live = f->objects[i].generation = f->objects[i].ns = 1;
        f->objects[i].rights = 3;
        memset(f->objects[i].data, 0x5a, 512);
        memset(f->objects[i].tags, 1, 32);
    }
    f->next.version = CN_ABI_VERSION; f->next.bytes = sizeof(f->next);
    f->next.translation = (struct cn_translation){1, 4096, 1, 1, 1, 1};
    struct cn_ops cn = {.validate_namespace = valid_namespace, .authorize = auth,
                        .step = step, .complete_service = complete};
    CHECK("engine create", !cn_engine_create(CN_BACKEND_IOCTL, &cn, f, &f->engine));
    CHECK("context create", !cn_create(f->engine, &f->owner, &f->next, &f->context));
    CHECK("session create", !ns_session_create(f->context, &f->owner, &ops, f, &f->session));
    f->epoch = 1;
    return f;
}
static void destroy(struct fixture *f)
{
    CHECK("all held resources released", f->held_count == f->released);
    ns_session_free(f->session); cn_engine_free(f->engine); free(f);
}
static void scalar(struct fixture *f, unsigned arg, cn_u64 value)
{ f->next.x[10 + arg] = (struct cn_register){.payload = {value, 0}}; }
static void capability(struct fixture *f, unsigned slot, unsigned object)
{ f->next.x[slot] = (struct cn_register){.payload = {object, f->objects[object].generation}, .flags = CN_REG_TAG}; }
static void request(struct fixture *f, cn_u64 nr)
{
    memset(f->next.x, 0, sizeof(f->next.x));
    f->next.x[17].payload[0] = NS_SERVICE_NATIVE;
    f->next.x[16].payload[0] = nr;
}
static void enter(struct fixture *f)
{
    struct cn_event event;
    CHECK("protected service entry", !cn_run(f->context, &f->owner, f->epoch,
          f->epoch == 1 ? CN_RESUME_RETRY : CN_RESUME_REPLY, 0, &event));
    f->epoch = event.context_epoch;
}
static struct ns_result dispatch(struct fixture *f)
{ struct ns_result r; CHECK("dispatch API", !ns_dispatch(f->session, f->epoch, &r)); return r; }
static void read_request(struct fixture *f, cn_u64 nr, unsigned fd, size_t size)
{ request(f, nr); scalar(f, 0, fd); capability(f, 11, 1); scalar(f, 2, size); enter(f); }
static void scalar_and_buffers(void)
{
    struct fixture *f = create();
    request(f, CAPSTONE_SYS_getpid); enter(f);
    CHECK("scalar result", dispatch(f).value == 123);
    CHECK("replay refused", dispatch(f).value == -ESTALE && f->issue_count == 1);
    request(f, UINT64_MAX); enter(f);
    CHECK("unknown refused", dispatch(f).value == -ENOSYS);
    for (cn_u64 nr = NS_MAP; nr <= NS_FUTEX; ++nr) {
        request(f, nr); enter(f); CHECK("grant backend absent", dispatch(f).value == -ENOSYS);
    }
    read_request(f, CAPSTONE_SYS_read, 3, 80);
    struct ns_result r = dispatch(f);
    CHECK("short I/O", r.value == 64 && r.bytes_out == 64 && f->objects[1].data[63] == 0x42 && f->objects[1].data[64] == 0x5a);
    CHECK("scalar output clears tags", f->objects[1].tags[0] == 0 && f->objects[1].tags[3] == 0 && f->objects[1].tags[4] == 1);
    read_request(f, CAPSTONE_SYS_write, 3, 64);
    CHECK("input snapshot", dispatch(f).value == 64 && f->written_bytes == 64 && f->written[63] == 0x42);
    request(f, CAPSTONE_SYS_clock_gettime); capability(f, 11, 2); enter(f);
    CHECK("timespec converted as RV64 words", dispatch(f).value == 0 && f->objects[2].data[0] == 7 && f->objects[2].data[8] == 9 && f->objects[2].data[16] == 0x5a);
    read_request(f, CAPSTONE_SYS_read, 3, 0);
    unsigned before = f->copies;
    CHECK("tagged zero checks U", dispatch(f).value == 0 && f->copies == before + 1);
    request(f, CAPSTONE_SYS_read); scalar(f, 0, 3); enter(f);
    CHECK("null zero length", dispatch(f).value == 0);
    request(f, CAPSTONE_SYS_read); scalar(f, 0, 3); scalar(f, 1, 999); enter(f);
    CHECK("forged zero refused", dispatch(f).value == -EFAULT);
    read_request(f, CAPSTONE_SYS_read, 3, 513);
    CHECK("bounds before side effects", dispatch(f).value == -EFAULT);
    f->objects[1].rights = 1;
    read_request(f, CAPSTONE_SYS_read, 3, 1);
    CHECK("write rights", dispatch(f).value == -EACCES);
    f->objects[1].rights = 3;
    request(f, CAPSTONE_SYS_getgroups); scalar(f, 0, UINT64_MAX); capability(f, 11, 1); enter(f);
    CHECK("scaled overflow", dispatch(f).value == -EOVERFLOW);
    struct ns_counters counters; ns_get_counters(f->session, &counters);
    printf("copy ledger: calls=%llu in=%llu out=%llu issued=%llu\n", (unsigned long long)counters.copy_calls,
        (unsigned long long)counters.bytes_in, (unsigned long long)counters.bytes_out, (unsigned long long)counters.issued);
    destroy(f);
}
static void lifetimes_and_restart(void)
{
    struct fixture *f = create();
    read_request(f, CAPSTONE_SYS_read, 3, 32);
    f->revoke_issue = 1;
    struct ns_result r = dispatch(f);
    CHECK("revoke during blocking issue", r.value == -EFAULT && !r.bytes_out && f->objects[1].data[0] == 0x5a);
    f->revoke_issue = 0;
    read_request(f, CAPSTONE_SYS_read, 3, 64);
    f->recycle_mid_copy = 1;
    r = dispatch(f);
    CHECK("partial I/O on recycled second chunk", r.value == 16 && r.bytes_out == 16 && r.cause == 25 && !r.restart);
    CHECK("recycled sentinel unchanged", f->objects[1].data[16] == 0x5a && f->objects[1].data[63] == 0x5a);
    f->force_completion = 1; f->completion = (struct ns_completion){-EINTR, 0, 1};
    read_request(f, CAPSTONE_SYS_read, 3, 4);
    r = dispatch(f); CHECK("restart without effects", r.value == -EINTR && r.restart && !r.bytes_out);
    f->completion.effects = 1;
    read_request(f, CAPSTONE_SYS_read, 3, 4);
    r = dispatch(f); CHECK("never restart after effects", r.value == -EPROTO && !r.restart);
    f->completion = (struct ns_completion){5, 1, 0};
    read_request(f, CAPSTONE_SYS_read, 3, 4);
    CHECK("oversized backend result", dispatch(f).value == -EPROTO);
    f->force_completion = 0; f->bad_copy = 1;
    read_request(f, CAPSTONE_SYS_read, 3, 4);
    CHECK("invalid copy result", dispatch(f).value == -EPROTO);
    f->bad_copy = 0;
    struct cn_capref ref;
    CHECK("valid current reference", !cn_get_capref(f->context, &f->owner, 11, &ref) &&
          !cn_authorize(f->context, &f->owner, &ref, 0, 1, 0));
    f->namespace_bad = 1;
    CHECK("invalid namespace refused", cn_authorize(f->context, &f->owner, &ref, 0, 1, 0) == -ESTALE);
    f->namespace_bad = 0;
    struct cn_owner foreign = f->owner; foreign.thread_id = 2;
    CHECK("foreign owner refused", cn_authorize(f->context, &foreign, &ref, 0, 1, 0) == -EPERM);
    CHECK("offset overflow refused", cn_authorize(f->context, &f->owner, &ref, UINT64_MAX, 2, 0) == -EOVERFLOW);
    destroy(f);
}
static void resources(void)
{
    struct fixture *f = create();
    const char *unsafe[] = {"/proc/self/mem", "/tmp/proc-alias", "/tmp/magic", "/tmp/mounted", "/tmp/../proc/self/mem"};
    for (unsigned i = 0; i < sizeof(unsafe)/sizeof(unsafe[0]); ++i) {
        request(f, CAPSTONE_SYS_openat); scalar(f, 0, (cn_u64)-100); capability(f, 11, 0);
        strcpy((char *)f->objects[0].data, unsafe[i]); enter(f);
        struct ns_result r = dispatch(f);
        CHECK("proc-memory and alias open refused", r.value == -EACCES);
        if (!i && r.value >= 0) {
            read_request(f, CAPSTONE_SYS_pread64, (unsigned)r.value, 4);
            r = dispatch(f);
            CHECK("proc-memory pread private sentinel", r.value < 0 && f->objects[1].data[0] != 0xcc);
        }
    }
    read_request(f, CAPSTONE_SYS_pread64, 4, 4);
    CHECK("inherited fd refused", dispatch(f).value == -EBADF);
    f->fd[3] = &f->private_memory;
    read_request(f, CAPSTONE_SYS_pread64, 3, 4);
    CHECK("descriptor replacement refused", dispatch(f).value == -EBADF);
    f->fd[3] = &f->normal; f->replace_fd = 1;
    read_request(f, CAPSTONE_SYS_read, 3, 4);
    CHECK("held file survives replacement", dispatch(f).value == 4 && f->objects[1].data[0] == 0x42);
    f->fd[3] = &f->normal; f->replace_fd = 0;
    request(f, CAPSTONE_SYS_ioctl); scalar(f, 0, 3); scalar(f, 1, 0xdead); enter(f);
    CHECK("hidden pointer command refused", dispatch(f).value == -ENOSYS);
    request(f, CAPSTONE_NR_IOCTL_BUF); scalar(f, 0, 3); enter(f);
    CHECK("buffered ioctl absent", dispatch(f).value == -ENOSYS);
    request(f, CAPSTONE_SYS_dup3); scalar(f, 0, 3); enter(f);
    CHECK("dup replacement absent", dispatch(f).value == -ENOSYS);
    request(f, CAPSTONE_SYS_getpid); enter(f); f->inherited_filter = 1;
    CHECK("policy refusal before scalar issue", dispatch(f).value == -EPERM);
    f->inherited_filter = 0; f->shared_fs = 1;
    request(f, CAPSTONE_SYS_getpid); enter(f);
    CHECK("shared fs refused", dispatch(f).value == -EPERM);
    f->shared_fs = 0;
    request(f, CAPSTONE_SYS_newfstatat); scalar(f, 0, (cn_u64)-100);
    capability(f, 11, 0); capability(f, 12, 0); strcpy((char *)f->objects[0].data, "/tmp/item"); enter(f);
    CHECK("overlapping argument snapshot", dispatch(f).value == 0 && f->objects[0].data[0] == 0x33);
    f->normal.permissions = 0;
    request(f, CAPSTONE_SYS_openat); capability(f, 11, 0); strcpy((char *)f->objects[0].data, "/tmp/item"); enter(f);
    CHECK("inode permission before open", dispatch(f).value == -EACCES);
    f->normal.permissions = 3;
    const char *safe[] = {"/tmp/item", "/usr/item", "/dev/null", "/dev/zero", "/dev/urandom"};
    for (unsigned i = 0; i < sizeof(safe)/sizeof(safe[0]); ++i) {
        request(f, CAPSTONE_SYS_openat); capability(f, 11, 0); strcpy((char *)f->objects[0].data, safe[i]); enter(f);
        CHECK("explicit root fixture", dispatch(f).value == 5);
    }
    request(f, CAPSTONE_SYS_close); scalar(f, 0, 3); enter(f);
    CHECK("close grant", dispatch(f).value == 0);
    read_request(f, CAPSTONE_SYS_read, 3, 1);
    CHECK("closed fd refused", dispatch(f).value == -EBADF);
    destroy(f);
}
static void vectors_and_missing_helpers(void)
{
    struct fixture *f = create();
    request(f, CAPSTONE_SYS_readv); scalar(f, 0, 3); capability(f, 11, 0); scalar(f, 2, 2);
    capability(f, 20, 1); capability(f, 21, 2); enter(f);
    CHECK("first nested reference", !cn_get_capref(f->context, &f->owner, 20, &f->vectors[0].base));
    CHECK("second nested reference", !cn_get_capref(f->context, &f->owner, 21, &f->vectors[1].base));
    f->vectors[0].length = 16; f->vectors[1].length = 24; f->mutate_vectors = 1;
    CHECK("iovec snapshot used for copyout", dispatch(f).value == 40 && f->objects[1].data[0] == 0x42 && f->objects[2].data[0] == 0x43);
    CHECK("nested refs released", f->iov_released == 1);
    enter(f); f->import_bad = 1;
    CHECK("forged nested payload refused", dispatch(f).value == -EFAULT);
    f->import_bad = 0;
    enter(f);
    CHECK("stale nested reference refused", dispatch(f).value == -ESTALE);
    scalar(f, 2, NS_MAX_IOV + 1); enter(f);
    CHECK("iovec count bounded", dispatch(f).value == -EINVAL);
    ns_session_free(f->session);
    struct ns_ops absent = ops; absent.copy = NULL;
    CHECK("absent U session", !ns_session_create(f->context, &f->owner, &absent, f, &f->session));
    read_request(f, CAPSTONE_SYS_write, 3, 4);
    CHECK("missing copy fails closed", dispatch(f).value == -ENOSYS);
    destroy(f);
}
static void exchange_adapter(void)
{
    struct fixture *f = create();
    struct ns_request r = {.nr = CAPSTONE_SYS_read, .tagged = 2};
    struct ns_result out;
    r.scalar[0] = 3; r.scalar[1] = 64; r.scalar[2] = 16;
    r.offsets[1] = 64;
    read_request(f, CAPSTONE_SYS_read, 3, 16);
    CHECK("exchange reference", !cn_get_capref(f->context, &f->owner, 11, &r.refs[1]));
    CHECK("exchange dispatch", !ns_dispatch_request(f->session, f->epoch, &r, &out));
    CHECK("exchange offset retained", out.value == 16 && f->objects[1].data[0] == 0x5a &&
          f->objects[1].data[63] == 0x5a && f->objects[1].data[64] == 0x42 &&
          f->objects[1].data[79] == 0x42 && f->objects[1].data[80] == 0x5a);
    CHECK("exchange replay", !ns_dispatch_request(f->session, f->epoch, &r, &out) && out.value == -ESTALE);
    enter(f);
    CHECK("exchange stale reference", !ns_dispatch_request(f->session, f->epoch, &r, &out) && out.value < 0);
    enter(f);
    CHECK("exchange fresh reference", !cn_get_capref(f->context, &f->owner, 11, &r.refs[1]));
    r.offsets[1] = UINT64_MAX;
    unsigned before = f->issue_count;
    CHECK("exchange offset overflow", !ns_dispatch_request(f->session, f->epoch, &r, &out) &&
          out.value == -EOVERFLOW && f->issue_count == before);
    destroy(f);
}
int main(void)
{
    scalar_and_buffers(); lifetimes_and_restart(); resources(); vectors_and_missing_helpers(); exchange_adapter();
    printf("native syscall checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
