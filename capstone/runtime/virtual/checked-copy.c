/* SPDX-License-Identifier: MIT */
#include <capstone/checked-copy.h>

struct cu_result cu_copy(const struct cu_ops *ops, void *ctx,
                         const struct cn_capref *ref, cn_u64 offset,
                         void *buffer, cn_u64 length, enum cu_direction dir,
                         cn_u64 page_bytes)
{
    struct cu_result r = {0, CU_REFUSED};
    cn_u64 address = 0, chunk, remaining, done;
    unsigned retries = 0;
    if (!ops || !ref || !ops->hold || !ops->release || !ops->enter ||
        !ops->check || !ops->copy || !ops->drain || !ops->leave ||
        !ops->resolve || !page_bytes || page_bytes > 4096 ||
        (page_bytes & (page_bytes - 1)) ||
        (dir != CU_FROM_DOMAIN && dir != CU_TO_DOMAIN) ||
        (length && !buffer)) return r;
    if (length > ~(cn_u64)0 - offset) { r.cause = CU_BOUNDS; return r; }
    r.cause = ops->hold(ctx, ref);
    if (r.cause) return r;
    do {
        ops->enter(ctx);
        /* First obtain a checked cursor; only the page-bounded chunk is
         * authorized below. Bounds must not be inferred from the address. */
        r.cause = ops->check(ctx, ref, offset + r.bytes_done, 0, dir, &address);
        chunk = page_bytes - (address & (page_bytes - 1));
        if (chunk > length - r.bytes_done) chunk = length - r.bytes_done;
        if (!r.cause)
            r.cause = ops->check(ctx, ref, offset + r.bytes_done, chunk, dir, &address);
        remaining = chunk;
        if (!r.cause && chunk)
            remaining = ops->copy(ctx, address, (char *)buffer + r.bytes_done, chunk, dir);
        ops->drain(ctx);
        ops->leave(ctx);
        if (r.cause) break;
        if (remaining > chunk) { r.cause = CU_REFUSED; break; }
        done = chunk - remaining;
        r.bytes_done += done;
        if (done) retries = 0;
        if (remaining) {
            if (++retries > 4) { r.cause = CU_REFUSED; break; }
            r.cause = ops->resolve(ctx, address + done, dir);
            if (r.cause) break;
            /* Resolution and scheduling can revoke an object without
             * changing its mapping: recheck the original capref on retry. */
        }
        if (r.bytes_done < length && ops->between) ops->between(ctx);
    } while (r.bytes_done < length);
    ops->release(ctx);
    return r;
}
