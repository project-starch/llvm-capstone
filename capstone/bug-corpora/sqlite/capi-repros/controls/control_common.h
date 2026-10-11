/* control_common.h -- the probes of the three lifetime controls (README.md here).
 *
 * Every access a control makes to a block goes through one of the two macros below, written
 * on ONE source line. The macro prints `PROBE <label> <op> <file>:<line> addr=... len=...`
 * BEFORE the access and `PROBE <label> done` after it, and the access itself is in the macro
 * body, not in a helper: clang gives macro-expanded code the line of the expansion, so a pc
 * that faults inside the access symbolizes (llvm-symbolizer, the image is built with -g) to
 * exactly the file:line the probe printed. A fault "at the probe" is therefore checkable two
 * ways: the last PROBE line has no `done`, and the fault pc resolves to that file:line.
 *
 * Included AFTER the amalgamation and repro322_common.h (see any control.c).
 */
#ifndef CONTROL_COMMON_H
#define CONTROL_COMMON_H

#include <stdint.h>
#include <stdio.h>

static unsigned long ctl_addr(const void *p) { return (unsigned long)(uintptr_t)p; }

static void ctl_say(const char *label, const char *op, const char *file, int line,
                    const void *p, unsigned n) {
  const char *base = strrchr(file, '/');
  printf("PROBE %s %s %s:%d addr=0x%lx len=%u\n", label, op, base ? base + 1 : file, line,
         ctl_addr(p), n);
}

/* Copy n bytes at ptr into buf (an unsigned char array), byte by byte through ptr. */
#define PROBE_READ(label, ptr, n, buf) do { const volatile unsigned char *ctl_p_ = (const volatile unsigned char *)(ptr); ctl_say((label), "read", __FILE__, __LINE__, (const void *)ctl_p_, (n)); for (unsigned ctl_i_ = 0; ctl_i_ < (n); ctl_i_++) (buf)[ctl_i_] = ctl_p_[ctl_i_]; printf("PROBE %s done\n", (label)); } while (0)

/* Store the n bytes of buf (an unsigned char array) at ptr, byte by byte through ptr. */
#define PROBE_WRITE(label, ptr, n, buf) do { volatile unsigned char *ctl_p_ = (volatile unsigned char *)(ptr); ctl_say((label), "write", __FILE__, __LINE__, (const void *)ctl_p_, (n)); for (unsigned ctl_i_ = 0; ctl_i_ < (n); ctl_i_++) ctl_p_[ctl_i_] = (buf)[ctl_i_]; printf("PROBE %s done\n", (label)); } while (0)

/* A recognisable pattern per role: byte i of role r is (r ^ i*7). */
static void ctl_fill(unsigned char *buf, unsigned n, unsigned char role) {
  for (unsigned i = 0; i < n; i++) buf[i] = (unsigned char)(role ^ (i * 7u));
}

static int ctl_failures;

/* One CHECK line per prediction the control tests; the runner requires every one `ok`. */
static void ctl_check(const char *what, int ok) {
  printf("CHECK %s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok) ctl_failures++;
}

/* The bounds a block's reference carries, read with the capability-query builtins. On the
 * release arm a memsys5 block is a pointer into the heap array, so these are the array's. */
static void ctl_bounds(const char *name, const void *p) {
  printf("BLOCK %s addr=0x%lx base=0x%lx end=0x%lx\n", name, ctl_addr(p),
         (unsigned long)__builtin_capstone_cap_get_base((void *)p),
         (unsigned long)__builtin_capstone_cap_get_end((void *)p));
}

#endif
