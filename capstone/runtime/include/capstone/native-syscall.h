/* SPDX-License-Identifier: MIT */
#ifndef CAPSTONE_NATIVE_SYSCALL_H
#define CAPSTONE_NATIVE_SYSCALL_H
#include <capstone/native-context.h>
#include <capstone/delegate.h>
#include <capstone/native-syscall-abi.h>

/* Experimental register ABI; not installed in the module or default SDK.
 * a7 selects this service, a6 is the number, a0..a5 retain capability tags.
 * Only a protected backend frame may be captured by the dispatcher. */
#define NS_MAP     UINT64_C(0xc0de1001)
#define NS_UNMAP   UINT64_C(0xc0de1002)
#define NS_PROTECT UINT64_C(0xc0de1003)
#define NS_NODES   UINT64_C(0xc0de1004)
#define NS_FUTEX   UINT64_C(0xc0de1005)
#define NS_MAX_BYTES (1024U * 1024U)
#define NS_MAX_PATH 4096U
#define NS_MAX_IOV 64U

enum ns_route { NS_REFUSE, NS_SCALAR, NS_FLAT, NS_VECTOR, NS_RESOURCE, NS_GRANT };
struct ns_descriptor {
    cn_u64 nr;
    const char *name;
    enum ns_route route;
    unsigned fd_mask;
};
const struct ns_descriptor *ns_descriptor_at(size_t);
const struct ns_descriptor *ns_descriptor(cn_u64);

enum ns_direction { NS_FROM_DOMAIN, NS_TO_DOMAIN };
struct ns_copy_result { cn_u64 bytes_done; unsigned cause; };
/* These references are produced ONLY by a trusted capability-aware snapshot
 * backend. A scalar serialization of cn_capref is never an authority import. */
struct ns_iov { struct cn_capref base; cn_u64 length; };
struct ns_buffer { void *data; size_t length; };
struct ns_invocation {
    const struct ns_descriptor *descriptor;
    cn_u64 scalar[6]; /* Pointer positions are zero, never scalar cursors. */
    struct ns_buffer buffer[6]; /* Private kernel-layout snapshots. */
    struct ns_buffer vector[NS_MAX_IOV];
    unsigned vector_count;
};
struct ns_completion {
    int64_t value;
    unsigned effects; /* Nonzero once an irreversible operation occurred. */
    unsigned restart; /* Only valid with value=-EINTR and effects=0. */
};
struct ns_result {
    int64_t value;
    cn_u64 bytes_out;
    unsigned cause, restart, effects;
};
struct ns_counters {
    cn_u64 copy_calls, bytes_in, bytes_out, iov_snapshots, issued;
};
struct ns_ops {
    /* U interface: authentication, exclusion, lifetime recheck, non-destructive
     * fault retry and tag clearing belong inside EACH call/chunk. */
    struct ns_copy_result (*copy)(void *, struct cn_context *, const struct cn_owner *,
        const struct cn_capref *, cn_u64 offset, void *kernel_buffer, size_t length,
        enum ns_direction);
    /* Snapshot nested capabilities with tags into protected references. The
     * provider owns these references until release_iov. No scalar fallback. */
    int (*snapshot_iov)(void *, struct cn_context *, const struct cn_owner *,
        const struct cn_capref *, size_t count, struct ns_iov *);
    void (*release_iov)(void *, struct cn_context *, struct ns_iov *, size_t count);
    /* Mandatory for every call: enforce credentials/seccomp semantics, command
     * policy and pin actual file identities. Called after ALL input snapshots;
     * held resources remain valid across blocking issue and copy-out. */
    int (*acquire)(void *, const struct ns_invocation *, void **held);
    struct ns_completion (*issue)(void *, const struct ns_invocation *, void *held);
    void (*release)(void *, void *held);
};
/* Trusted exchange adapter: offsets refer to actual protected register refs.
 * This structure is never accepted from an ioctl or copied from userspace. */
struct ns_request {
    cn_u64 nr, scalar[6], offsets[6];
    struct cn_capref refs[6];
    unsigned tagged, nulls;
};
struct ns_session;
/* Shared dispatcher for the executable double and experimental kernel adapter.
 * Providers supply the authority checks and OS operations. Serialized calls
 * only; one session per protected context for its entire lifetime. */
int ns_session_create(struct cn_context *, const struct cn_owner *, const struct ns_ops *,
                      void *, struct ns_session **);
void ns_session_free(struct ns_session *);
int ns_dispatch(struct ns_session *, cn_u64 epoch, struct ns_result *);
int ns_dispatch_request(struct ns_session *, cn_u64 epoch,
                        const struct ns_request *, struct ns_result *);
void ns_get_counters(const struct ns_session *, struct ns_counters *);
#endif
