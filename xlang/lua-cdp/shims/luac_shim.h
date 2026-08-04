/* Declarations for the Lua-CDP Capstone shims.
 *
 * These shims are the SAME artifact-shape as ../../cheri/shims/rlua_userdata_uaf.c:
 * a standalone main() that models one case's memory lifecycle with malloc/free
 * and reports on the survival path. They are #included into
 * ../../capstone/xlang_shim_domain.c (ROW_SRC) and linked against
 * ../../capstone/mock_mruby_capstone.c.
 *
 * WHERE EACH SYMBOL COMES FROM — and why that is the whole measurement:
 *   malloc/free   ../../capstone/mock_mruby_capstone.c, on top of rof_malloc/
 *                 rof_free. A shim's own free() therefore REVOKES the block; the
 *                 later stale deref is what the mechanism must police. If these
 *                 were libc's, every row would run clean and report MISS while
 *                 measuring nothing.
 *   memcpy/memset ../../capstone/xlang_shim_domain.c (byte-wise, tag-agnostic).
 *   abort         xlang_shim_domain.c — routed to the loud XLANG-INVALID path so
 *                 a failed precondition can never be read as a MISS.
 *   mock_report   xlang_shim_domain.c — writes the survival marker into the host
 *                 payload region. A row that FAULTS never reaches it, which is
 *                 what makes FAULT and MISS distinguishable from the host side.
 */
#ifndef LUAC_SHIM_H
#define LUAC_SHIM_H

#include <stddef.h>

void *malloc(size_t n);
void  free(void *p);
void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
void  abort(void);
void  mock_report(const char *row, const char *what);

#endif /* LUAC_SHIM_H */
