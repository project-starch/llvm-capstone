/* SPDX-License-Identifier: MIT */
#include "syscall_arch.h"
long native_probe(void *a, void *b, void *c, void *d, void *e, void *f)
{ return __syscall6(63, a, b, c, d, e, f); }
