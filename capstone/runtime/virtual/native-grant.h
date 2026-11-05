#ifndef CAPSTONE_NATIVE_GRANT_H
#define CAPSTONE_NATIVE_GRANT_H

#include <linux/types.h>
#include <linux/ioctl.h>

/* Draft interface owned by the fault/mapping lane. Scalar identities select
 * protected records; they never authenticate a capability supplied by a task. */
#define CV_GRANT_ABI_VERSION 1
enum cv_grant_operation {
    CV_GRANT_MAP = 1,
    CV_GRANT_UNMAP = 2,
    CV_GRANT_PROTECT = 3,
    CV_GRANT_REMAP = 4,
};
enum cv_fault_class {
    CV_FAULT_DENIED = 0,
    CV_FAULT_PAGE_READ = 1,
    CV_FAULT_PAGE_WRITE = 2,
    CV_FAULT_PAGE_EXEC = 3,
};
static inline enum cv_fault_class cv_fault_classify(__u64 event, __u64 cause)
{
    if (event != 4) return CV_FAULT_DENIED;
    switch (cause) {
    case 12: return CV_FAULT_PAGE_EXEC;
    case 13: return CV_FAULT_PAGE_READ;
    case 15: return CV_FAULT_PAGE_WRITE;
    default: return CV_FAULT_DENIED;
    }
}

/* Input-only prototype ioctl: successful commit has no fallible user copy.
 * namespace identity is the opened context, checked against current->mm.
 * id is never reused within that context. Exact address/length must match.
 * Nonzero flags/reserved fields and every unimplemented operation fail closed.
 * Injection is supported only in a CV_GRANT_TXN test module. */
struct cv_grant_transaction {
    __u32 version, operation;
    __u64 id, address, bytes;
    __u32 inject, flags;
    __u64 reserved;
};
#define CV_GRANT_TRANSACTION _IOW('V', 40, struct cv_grant_transaction)

enum cv_grant_injection {
    CV_GRANT_INJECT_NONE = 0,
    CV_GRANT_INJECT_LOCK = 1,
    CV_GRANT_INJECT_VALIDATE = 2,
    CV_GRANT_INJECT_TREE = 3,
    CV_GRANT_INJECT_DETACH = 4,
    CV_GRANT_INJECT_RETIRE = 5,
};

#ifdef __KERNEL__
struct mm_struct;
/* Caller holds its namespace lock before this helper takes mmap_write_lock.
 * commit must return an error without changing authority, or return zero
 * after irrevocable retirement. It must not acquire mmap_lock recursively.
 * After zero, the helper cannot fail or release exclusion before unmapping.
 * Only exact, private, anonymous, unlocked, non-userfaultfd VMAs are accepted. */
extern int capstone_grant_unmap(struct mm_struct *mm, unsigned long address,
                              unsigned long bytes, int (*commit)(void *),
                              void *opaque, unsigned int inject);
#endif
#endif
