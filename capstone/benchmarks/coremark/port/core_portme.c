#include "coremark.h"
#include "coremark_hostcall.h"

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
 * The compiler-generated va_list stores the argument-pointer as a SCALAR
 * integer (sd, not stc).  When reloaded via ld, the register tag is 0 and
 * any subsequent memory dereference crashes in cap_mem mode.
 *
 * Fix: the real entry point `ee_printf` is an assembly trampoline
 * (ee_printf_asm.S) that receives a0=fmt and a1-a7=varargs in registers and
 * calls `ee_printf_impl(fmt, a1..a7)` directly — no va_list involved.
 * `ee_printf_impl` selects the Nth argument via a fixed-offset if-chain
 * (pick_arg) so no variable-index capability arithmetic is needed.
 *
 * DELIN is applied to fmt, pay, and %s string args because all three are
 * LINEAR capabilities on Capstone PureCap.  The host flushes the
 * accumulated buffer after HC_V0_RET_DONE.
 *
 * All local counters use unsigned long (i64) to avoid sext-from-i32
 * FrameIndex loads that the Capstone ISel cannot match.
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

/*
 * Return the idx-th variadic argument as a void capability.
 * Each branch loads from a FIXED s0-relative RSA slot (no variable-index
 * capability arithmetic), so the base capability is never consumed.
 */
static void *pick_arg(unsigned long idx,
                      void *a1, void *a2, void *a3, void *a4,
                      void *a5, void *a6, void *a7) {
  if (idx == 0) return a1;
  if (idx == 1) return a2;
  if (idx == 2) return a3;
  if (idx == 3) return a4;
  if (idx == 4) return a5;
  if (idx == 5) return a6;
  return a7;
}

/*
 * Called from the ee_printf assembly trampoline with a0=fmt and
 * a1-a7 = the first seven variadic arguments (as capability registers).
 */
int ee_printf_impl(const char *fmt,
                   void *a1, void *a2, void *a3, void *a4,
                   void *a5, void *a6, void *a7) {
  CAPSTONE_DELIN(fmt);
  if (!hc_metadata || !hc_payload)
    return 0;
  char *pay = (char *)hc_payload;
  CAPSTONE_DELIN(pay);
  unsigned long off = hc_metadata->length;
  unsigned long max = HC_REGION_SIZE;
  unsigned long idx = 0;

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
    if (spec == 'l') { spec = *fmt++; }  /* eat 'l' prefix */
    if (spec == 'd' || spec == 'i') {
      long v = (long)(unsigned long)(uintptr_t)pick_arg(idx++, a1, a2, a3, a4, a5, a6, a7);
      if (v < 0) { if (off < max - 1) pay[off++] = '-'; v = -v; }
      off = hcp_uint(pay, off, max, (unsigned long)v, 10, width, pad);
    } else if (spec == 'u') {
      unsigned long v = (unsigned long)(uintptr_t)pick_arg(idx++, a1, a2, a3, a4, a5, a6, a7);
      off = hcp_uint(pay, off, max, v, 10, width, pad);
    } else if (spec == 'x') {
      unsigned long v = (unsigned long)(uintptr_t)pick_arg(idx++, a1, a2, a3, a4, a5, a6, a7);
      off = hcp_uint(pay, off, max, v, 16, width, pad);
    } else if (spec == 's') {
      /* String arg is a LINEAR capability; DELIN before multi-char loop. */
      const char *s = (const char *)pick_arg(idx++, a1, a2, a3, a4, a5, a6, a7);
      CAPSTONE_DELIN(s);
      if (!s) s = "(null)";
      while (*s && off < max - 1)
        pay[off++] = *s++;
    } else if (spec == '%') {
      if (off < max - 1) pay[off++] = '%';
    }
  }

  hc_metadata->length = off;
  return 0;
}
