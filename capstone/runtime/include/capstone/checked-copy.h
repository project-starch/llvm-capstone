/* SPDX-License-Identifier: MIT */
#ifndef CAPSTONE_CHECKED_COPY_H
#define CAPSTONE_CHECKED_COPY_H
#include <capstone/native-context.h>

enum cu_direction { CU_FROM_DOMAIN, CU_TO_DOMAIN };
struct cu_result { cn_u64 bytes_done; cn_u32 cause; };
#define CU_TAG 24U
#define CU_LIFETIME 25U
#define CU_TYPE 26U
#define CU_RIGHTS 27U
#define CU_BOUNDS 28U
#define CU_REFUSED 29U

/* Trusted backend only. enter/leave exclude scheduling AND all revokers on
 * the supported single hart and disable automatic page fault handling.
 * check obtains the address from the protected register, never from a client
 * scalar. It must authenticate ctx/ref, namespace, tag, type, rights, bounds
 * and lifetime even for length zero. copy returns bytes NOT copied; resolve
 * may sleep, but must never write a dummy byte. Holds cover the whole call.
 * drain precedes leave on every path, including partial and denied copies. */
struct cu_ops {
    cn_u32 (*hold)(void *, const struct cn_capref *);
    void (*release)(void *);
    void (*enter)(void *);
    cn_u32 (*check)(void *, const struct cn_capref *, cn_u64 offset,
                    cn_u64 length, enum cu_direction, cn_u64 *address);
    cn_u64 (*copy)(void *, cn_u64 address, void *kernel_buffer,
                   cn_u64 length, enum cu_direction);
    void (*drain)(void *);
    void (*leave)(void *);
    cn_u32 (*resolve)(void *, cn_u64 address, enum cu_direction);
    void (*between)(void *); /* optional reschedule / deterministic test hook */
};

/* page_bytes is a power of two, <= 4096; retry budget prevents a non-progress
 * fault from spinning forever. Unsupported/malformed backends fail closed. */
struct cu_result cu_copy(const struct cu_ops *, void *, const struct cn_capref *,
                         cn_u64 offset, void *kernel_buffer, cn_u64 length,
                         enum cu_direction, cn_u64 page_bytes);
#endif
