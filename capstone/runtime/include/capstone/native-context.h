/* SPDX-License-Identifier: MIT */
#ifndef CAPSTONE_NATIVE_CONTEXT_H
#define CAPSTONE_NATIVE_CONTEXT_H

/* Trusted context/event interface proposal, version 1. This is not a userspace
 * ioctl wire format and does not authorize importing scalar bytes as tags. */
#ifdef __KERNEL__
#include <linux/types.h>
typedef __u64 cn_u64;
typedef __u32 cn_u32;
#else
#include <stdint.h>
typedef uint64_t cn_u64;
typedef uint32_t cn_u32;
#endif

#define CN_ABI_VERSION 1U
#define CN_REGISTER_COUNT 32U
#define CN_CAP_CSR_COUNT 4U /* PCC, CTVEC, CEPC, CSCRATCH */
#define CN_MAX_SIGNAL_DEPTH 8U
#define CN_FEATURE_FP (1ULL << 0)
#define CN_FEATURE_EXACT_BOUNDS (1ULL << 1)
#define CN_FEATURES_V1 (CN_FEATURE_FP | CN_FEATURE_EXACT_BOUNDS)
#define CN_REG_TAG 1U
#define CN_REG_RAW_TAG 2U
#define CN_REG_EXACT_BOUNDS 4U
#define CN_REG_FLAGS (CN_REG_TAG | CN_REG_RAW_TAG | CN_REG_EXACT_BOUNDS)

/* A backend must preserve both 128 payload bits even when untagged. Exact
 * bounds and opaque raw-move cargo are separate state, never reconstructed
 * from a scalar cursor. Unsupported encodings/features fail before entry. */
struct cn_register {
    cn_u64 payload[2];
    cn_u64 exact_base, exact_end;
    cn_u32 flags;
    cn_u32 reserved;
};
struct cn_owner {
    cn_u64 thread_id;
    cn_u64 address_space_id;
    cn_u64 namespace_id;
    cn_u64 namespace_epoch;
};
struct cn_translation {
    cn_u64 satp;
    cn_u64 root_pa;
    cn_u64 translation_epoch;
    cn_u64 address_space_id;
    cn_u64 namespace_id;
    cn_u64 namespace_epoch;
};
struct cn_frame {
    cn_u32 version, bytes;
    cn_u64 features;
    struct cn_register x[CN_REGISTER_COUNT];
    struct cn_register ccsr[CN_CAP_CSR_COUNT];
    cn_u64 f[CN_REGISTER_COUNT]; /* IEEE payloads, including NaN payloads */
    cn_u64 pc, status, fcsr;
    struct cn_translation translation;
    cn_u64 reserved[4];
};

enum cn_backend { CN_BACKEND_IOCTL = 1, CN_BACKEND_ARCH_RETURN = 2 };
enum cn_event_kind {
    CN_EVENT_QUANTUM = 1, CN_EVENT_SERVICE, CN_EVENT_PAGE_FAULT,
    CN_EVENT_CAP_FAULT, CN_EVENT_SIGNAL, CN_EVENT_EXIT, CN_EVENT_REFUSED
};
enum cn_fault_class {
    CN_FAULT_NONE, CN_FAULT_TRANSLATION, CN_FAULT_TAG, CN_FAULT_TYPE,
    CN_FAULT_RIGHTS, CN_FAULT_BOUNDS, CN_FAULT_LIFETIME, CN_FAULT_NAMESPACE,
    CN_FAULT_BACKEND
};
enum cn_resume { CN_RESUME_RETRY, CN_RESUME_REPLY };
struct cn_event {
    cn_u32 version, bytes, kind, fault_class;
    cn_u64 context_epoch, sequence;
    cn_u64 fault_pc, restart_pc, cause, address;
};
struct cn_capref {
    cn_u64 context_id, context_epoch;
    cn_u32 frame_slot, reserved;
};
struct cn_signal_token {
    cn_u64 context_id, nonce;
    cn_u32 depth, reserved;
};
struct cn_counters {
    cn_u64 entries, evictions, restorations, bytes_moved;
};

/* All functions are called by trusted kernel/backend code. `actor` must come
 * from the current task, not an ioctl argument. Consumers never receive a
 * writable context frame. The reference implementation is in tests/native-context. */
struct cn_engine;
struct cn_context;
struct cn_ops {
    int (*validate_namespace)(void *, const struct cn_owner *, const struct cn_translation *);
    int (*clone_namespace)(void *, const struct cn_owner *, const struct cn_translation *,
                           const struct cn_owner *, struct cn_translation *);
    void (*abort_clone)(void *, const struct cn_owner *);
    int (*authorize)(void *, const struct cn_owner *, const struct cn_register *,
                     cn_u64 offset, cn_u64 length, int write);
    int (*step)(void *, struct cn_frame *, struct cn_event *);
    int (*complete_service)(void *, struct cn_frame *, cn_u64 scalar_reply);
};
int cn_engine_create(enum cn_backend, const struct cn_ops *, void *, struct cn_engine **);
void cn_engine_free(struct cn_engine *);
int cn_create(struct cn_engine *, const struct cn_owner *, const struct cn_frame *, struct cn_context **);
int cn_run(struct cn_context *, const struct cn_owner *actor, cn_u64 epoch,
           enum cn_resume, cn_u64 reply, struct cn_event *);
int cn_get_epoch(struct cn_context *, const struct cn_owner *actor, cn_u64 *);
int cn_snapshot(struct cn_context *, const struct cn_owner *actor, struct cn_frame *);
int cn_get_capref(struct cn_context *, const struct cn_owner *actor, unsigned slot, struct cn_capref *);
int cn_authorize(struct cn_context *, const struct cn_owner *actor, const struct cn_capref *,
                 cn_u64 offset, cn_u64 length, int write);
int cn_scalar_write(struct cn_context *, const struct cn_owner *actor, cn_u64 epoch,
                    unsigned slot, cn_u64 value);
int cn_signal_push(struct cn_context *, const struct cn_owner *actor, cn_u64 epoch,
                   const struct cn_frame *trusted_handler, struct cn_signal_token *);
int cn_signal_return(struct cn_context *, const struct cn_owner *actor, cn_u64 epoch,
                     const struct cn_signal_token *);
int cn_clone(struct cn_context *, const struct cn_owner *actor,
             const struct cn_owner *child_owner, struct cn_context **);
int cn_destroy(struct cn_context *, const struct cn_owner *actor);
void cn_get_counters(const struct cn_engine *, struct cn_counters *);

#endif
