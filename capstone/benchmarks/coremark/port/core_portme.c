#include "coremark.h"
#include "coremark_hostcall.h"
#include <stdarg.h>

#ifndef ITERATIONS
#define ITERATIONS 10
#endif

#ifndef COREMARK_DEFAULT_EXECS
#define COREMARK_DEFAULT_EXECS 0
#endif

#if VALIDATION_RUN
volatile ee_s32 seed1_volatile = 0x3415;
volatile ee_s32 seed2_volatile = 0x3415;
volatile ee_s32 seed3_volatile = 0x66;
#elif PERFORMANCE_RUN
volatile ee_s32 seed1_volatile = 0x0;
volatile ee_s32 seed2_volatile = 0x0;
volatile ee_s32 seed3_volatile = 0x66;
#else
volatile ee_s32 seed1_volatile = 0x8;
volatile ee_s32 seed2_volatile = 0x8;
volatile ee_s32 seed3_volatile = 0x8;
#endif

volatile ee_s32 seed4_volatile = ITERATIONS;
volatile ee_s32 seed5_volatile = COREMARK_DEFAULT_EXECS;

ee_u32 default_num_contexts = 1;

static CORE_TICKS g_start_time;
static CORE_TICKS g_stop_time;

void *portable_malloc(ee_size_t size) {
  (void)size;
  return 0;
}

void portable_free(void *p) {
  (void)p;
}

void start_time(void) {
  g_start_time = 0;
  g_stop_time = 0;
}

void stop_time(void) {
  g_stop_time = 10000;
}

CORE_TICKS get_time(void) {
  return g_stop_time - g_start_time;
}

secs_ret time_in_secs(CORE_TICKS ticks) {
  return (secs_ret)(ticks / 1000u);
}

void portable_init(core_portable *p, int *argc, char *argv[]) {
  (void)argc;
  (void)argv;
  /* Skip sizeof(ee_ptr_int) == sizeof(void*) check: on Capstone PureCap
   * uintptr_t is 8 bytes (cursor) while void* is 16 bytes (capability).
   * The mismatch is intentional — ee_ptr_int holds the cursor/address part. */
  if (sizeof(ee_u32) != 4)
    return;
  p->portable_id = 1;
}

void portable_fini(core_portable *p) {
  p->portable_id = 0;
}

/*
 * ee_printf with format substitution.
 *
 * Uses the standard C va_list.  The Capstone backend now lowers va_start/va_arg
 * with capability operations (stc/ldc and a 16-byte cincoffset stride), so the
 * va_list pointer keeps its tag and each variadic argument is read from its own
 * capability slot — no assembly trampoline is needed.
 *
 * DELIN is applied to fmt, pay, and %s string args because all three are
 * LINEAR capabilities on Capstone PureCap.  The host flushes the
 * accumulated buffer after HC_V0_RET_DONE.
 *
 * All local counters use unsigned long (i64) to avoid sext-from-i32
 * FrameIndex loads that the Capstone ISel cannot match.  Integer arguments are
 * read with unsigned-long-width va_arg slots; %f is consumed as an integer slot
 * (this freestanding port never formats floating point, matching prior
 * behavior) so no soft-float support is required.
 */
extern volatile struct hostcall_v0 *hc_metadata;
extern volatile char               *hc_payload;

#define CAPSTONE_DELIN(rd) \
    __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static unsigned long hcp_uint(char *pay, unsigned long off, unsigned long max,
                               unsigned long val, unsigned long base,
                               unsigned long width, char pad) {
  char tmp[32];
  unsigned long n = 0;
  if (val == 0) {
    tmp[n++] = '0';
  } else {
    while (val > 0) {
      unsigned long r = val % base;
      tmp[n++] = (r < 10) ? (char)('0' + r) : (char)('a' + r - 10);
      val /= base;
    }
  }
  unsigned long i;
  for (i = n; i < width && off < max - 1; i++)
    pay[off++] = pad;
  while (n > 0 && off < max - 1)
    pay[off++] = tmp[--n];
  return off;
}

int ee_printf(const char *fmt, ...) {
  /* Same hazard as S-02 in sqlite_capstone_domain.c's output_text(), proven on silicon
     2026-08-09: on the gp-captable ABI these capabilities are reached through the
     cap-table and arrive NONLIN, and DELIN on a non-linear capability raises
     UNEXPECTED_CAP_TYPE on the RTL -- which WEDGES rather than traps. QEMU's
     helper_csdelin returns early, hiding it entirely under emulation.
     UNTESTED HERE: this file has not been re-run on the board since the guard was added.
     It is the same construct on the same ABI, but that is an inference, not a measurement. */
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  CAPSTONE_DELIN(fmt);
#endif
  if (!hc_metadata || !hc_payload)
    return 0;
  char *pay = (char *)hc_payload;
#ifndef CAPSTONE_GP_CAPTABLE_ABI
  CAPSTONE_DELIN(pay);
#endif
  unsigned long off = hc_metadata->length;
  unsigned long max = HC_REGION_SIZE;

  va_list ap;
  va_start(ap, fmt);

  while (*fmt) {
    if (*fmt != '%') {
      if (off < max - 1) pay[off++] = *fmt;
      fmt++;
      continue;
    }
    fmt++;  /* skip '%' */
    unsigned long width = 0;
    char pad = ' ';
    if (*fmt == '0') { pad = '0'; fmt++; }
    while (*fmt >= '0' && *fmt <= '9') { width = width * 10 + (unsigned long)(*fmt - '0'); fmt++; }
    char spec = *fmt++;
    int is_long = 0;
    if (spec == 'l') { is_long = 1; spec = *fmt++; }  /* eat 'l' prefix */
    if (spec == 'd' || spec == 'i') {
      long v = is_long ? va_arg(ap, long) : (long)va_arg(ap, int);
      if (v < 0) { if (off < max - 1) pay[off++] = '-'; v = -v; }
      off = hcp_uint(pay, off, max, (unsigned long)v, 10, width, pad);
    } else if (spec == 'u') {
      unsigned long v =
          is_long ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
      off = hcp_uint(pay, off, max, v, 10, width, pad);
    } else if (spec == 'x') {
      unsigned long v =
          is_long ? va_arg(ap, unsigned long) : (unsigned long)va_arg(ap, unsigned int);
      off = hcp_uint(pay, off, max, v, 16, width, pad);
    } else if (spec == 's') {
      /* String arg is a LINEAR capability; DELIN before multi-char loop. */
      const char *s = va_arg(ap, const char *);
      CAPSTONE_DELIN(s);
      if (!s) s = "(null)";
      while (*s && off < max - 1)
        pay[off++] = *s++;
    } else if (spec == '%') {
      if (off < max - 1) pay[off++] = '%';
    } else if (spec == 'f') {
      /* Freestanding port: no soft-float.  Consume the argument's slot as an
       * integer so the va_list cursor stays aligned, but emit nothing. */
      (void)va_arg(ap, unsigned long);
    }
  }

  va_end(ap);
  hc_metadata->length = off;
  return 0;
}
