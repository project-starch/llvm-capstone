/* Freestanding <stdlib.h> for the Capstone domain build, shadowing glibc's via
 * -I. Two shims (rlua_userdata_uaf, libpulse_iterator_uaf) allocate directly
 * rather than through the mock, so they include <stdlib.h>.
 *
 * WHY SHADOW IT. glibc's headers assume 8-byte pointers and do not survive a
 * 16-byte-pointer target: bits/floatn.h wants __float128, and bits/types.h
 * redefines __int64_t as `long long` against the target's `long`. Same class as
 * <stdio.h>, whose struct _IO_FILE sizes padding as
 * `12*sizeof(int) - 5*sizeof(void*)` and goes negative.
 *
 * WHY IT MATTERS BEYOND COMPILING. malloc/free here are NOT libc's -- they are
 * implemented in mock_mruby_capstone.c on top of rof_malloc/rof_free, so a
 * shim's own free() REVOKES. If they were plain allocations those two rows
 * would run clean and report MISS while measuring nothing at all, which is the
 * failure mode this file exists to prevent.
 */
#ifndef XLANG_CAPSTONE_STDLIB_H
#define XLANG_CAPSTONE_STDLIB_H

#include <stddef.h>

void *malloc(size_t n);
void  free(void *p);
void  abort(void);

#endif
