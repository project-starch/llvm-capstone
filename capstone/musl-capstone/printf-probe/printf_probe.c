/* Formatted output and the libc allocator, inside a pure-capability domain.
 *
 * Two things are under test and they are deliberately in one probe, because
 * neither is much use without the other and both are new to libc-capstone.a:
 *
 *   1. vfprintf, generated from musl's own by narrowing `long double` to
 *      `double` (libc-ext/gen-vfprintf-double.py). Everything in stdio funnels
 *      through it, so this is what makes printf/snprintf/fprintf/puts exist at
 *      all on this target.
 *   2. malloc/free/realloc/calloc from libc-ext/malloc.c, now in the archive
 *      rather than copied into each workload.
 *
 * WHY THE EXPECTED STRINGS ARE HARD-CODED AND WHERE THEY COME FROM. They were
 * produced by running the SAME format strings against glibc on the host. So a
 * mismatch means the domain's libc disagrees with a reference libc, not with my
 * expectation of what printf does. Cases 5 and 8 are the ones that matter for
 * the narrowing: %.17g of 0.1 needs all 17 significant digits right, and
 * %.0f of 2.5 -> "2" and %.1f of -0.25 -> "-0.2" need round-half-to-even. A
 * formatter that lost precision in the transform fails those and passes the
 * easy integer cases.
 *
 * WHY STAGE MARKERS GO THROUGH SAY() AND NOT printf. printf is the thing being
 * tested. If it is broken, a probe that reports through it says nothing at all,
 * which is indistinguishable from a wedge. SAY writes through the raw hostcall
 * that musl-hello already proved.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);
#define SAY(s) __capstone_hc_write(1, (s), sizeof(s) - 1)

/* TRACE, for localizing a fault rather than for the pass/fail run.
 *
 * A capability fault inside the domain does not return an error: QEMU asserts
 * and dies, so nothing after it runs and no marker sequence spanning several
 * domains would survive either. The last TRACE line the host printed is
 * therefore the bisection point, and it costs one boot instead of a ladder of
 * images. Off by default so the real probe's output stays readable.  */
#ifdef PRINTF_PROBE_TRACE
#define TRACE(s) SAY("PRINTF TRACE: " s "\n")
#else
#define TRACE(s) ((void)0)
#endif

/* NEGATIVE CONTROL. A probe that has only ever passed is an unproven probe, and
 * this one is now the gate on two new pieces of libc. Built with
 * -DPRINTF_PROBE_NEGATIVE_CONTROL it corrupts one expectation and one block
 * stamp, so both arms of the check -- the format comparison and the allocator's
 * overlap detection -- have to report and the run must FAIL. Run it whenever
 * either check is changed; passing under this flag means the check is inert. */
#ifdef PRINTF_PROBE_NEGATIVE_CONTROL
#define WANT(s) "wrong-" s
#define STAMP(i) ((unsigned char)((i) + 1 + ((i) == 7)))
#else
#define WANT(s) s
#define STAMP(i) ((unsigned char)((i) + 1))
#endif

static int failures;

static void say_str(const char *s) { __capstone_hc_write(1, s, strlen(s)); }

/* Report BOTH strings on a mismatch. "case 5 failed" costs another run to find
   out what it printed; "want X got Y" is the whole diagnosis. */
static void expect(const char *label, const char *got, const char *want) {
  if (strcmp(got, want) == 0)
    return;
  failures++;
  say_str("PRINTF FAIL ");
  say_str(label);
  say_str(": want |");
  say_str(want);
  say_str("| got |");
  say_str(got);
  say_str("|\n");
}

static void check_formats(void) {
  char b[128];

  TRACE("before literal-only snprintf");
  snprintf(b, sizeof b, "plain");
  expect("0 no directives", b, WANT("plain"));

  TRACE("before integers");
  snprintf(b, sizeof b, "%d %u %x %o %c %s %%", -42, 42u, 42, 42, 'A', "hi");
  expect("1 integers", b, WANT("-42 42 2a 52 A hi %"));

  TRACE("before width+precision float");
  snprintf(b, sizeof b, "%08.3f", 3.14159);
  expect("2 width+precision", b, WANT("0003.142"));

  TRACE("before exponent");
  snprintf(b, sizeof b, "%e", 12345.678);
  expect("3 exponent", b, WANT("1.234568e+04"));

  TRACE("before %g");
  snprintf(b, sizeof b, "%g %g", 0.0001, 0.00001);
  expect("4 %g switchover", b, WANT("0.0001 1e-05"));

  TRACE("before %.17g");
  snprintf(b, sizeof b, "%.17g", 0.1);
  expect("5 full double precision", b, WANT("0.10000000000000001"));

  TRACE("before flags");
  snprintf(b, sizeof b, "%-6.3s|%+d|% d", "abcdef", 7, 7);
  expect("6 flags", b, WANT("abc   |+7| 7"));

  TRACE("before length modifiers");
  snprintf(b, sizeof b, "%ld %lld %zu", -1L, 1234567890123LL, (size_t)99);
  expect("7 length modifiers", b, WANT("-1 1234567890123 99"));

  TRACE("before round half to even");
  snprintf(b, sizeof b, "%.0f %.1f", 2.5, -0.25);
  expect("8 round half to even", b, WANT("2 -0.2"));
  TRACE("formats done");
}

/* THE ALLOCATOR CHECK IS ABOUT OVERLAP, not about malloc returning non-null.
 * Every block is stamped with its own index; if first-fit, splitting or
 * coalescing ever hands the same bytes out twice, some block reads back a
 * neighbour's stamp. Freeing every second block and allocating again is what
 * exercises the split/coalesce paths rather than a plain bump. */
#define BLOCKS 32

static void check_allocator(void) {
  unsigned char *p[BLOCKS];
  size_t n[BLOCKS];

  TRACE("before first malloc");

  for (int i = 0; i < BLOCKS; i++) {
    n[i] = 16 + (size_t)i * 37;
    p[i] = malloc(n[i]);
    if (!p[i]) {
      failures++;
      SAY("PRINTF FAIL: malloc returned null\n");
      return;
    }
    memset(p[i], STAMP(i), n[i]);
  }
  for (int i = 0; i < BLOCKS; i += 2) {
    free(p[i]);
    p[i] = 0;
  }
  for (int i = 0; i < BLOCKS; i += 2) {
    n[i] = 8 + (size_t)i * 11;
    p[i] = malloc(n[i]);
    if (!p[i]) {
      failures++;
      SAY("PRINTF FAIL: malloc returned null after free\n");
      return;
    }
    memset(p[i], STAMP(i), n[i]);
  }
  for (int i = 0; i < BLOCKS; i++)
    for (size_t k = 0; k < n[i]; k++)
      if (p[i][k] != (unsigned char)(i + 1)) {
        failures++;
        SAY("PRINTF FAIL: allocator handed out overlapping blocks\n");
        return;
      }
  for (int i = 0; i < BLOCKS; i++)
    free(p[i]);

  TRACE("before realloc");
  /* realloc must PRESERVE, and grow past what the block can hold in place. */
  char *s = malloc(8);
  if (!s) {
    failures++;
    SAY("PRINTF FAIL: malloc(8)\n");
    return;
  }
  memcpy(s, "1234567", 8);
  s = realloc(s, 4096);
  if (!s || strcmp(s, "1234567") != 0) {
    failures++;
    SAY("PRINTF FAIL: realloc lost the contents\n");
    return;
  }
  free(s);

  TRACE("before calloc");
  unsigned char *z = calloc(64, 7);
  if (!z) {
    failures++;
    SAY("PRINTF FAIL: calloc\n");
    return;
  }
  for (int i = 0; i < 64 * 7; i++)
    if (z[i] != 0) {
      failures++;
      SAY("PRINTF FAIL: calloc did not zero\n");
      free(z);
      return;
    }
  free(z);
}

int capstone_main(void) {
  SAY("PRINTF S1: entered\n");

  check_formats();
  SAY("PRINTF S2: snprintf cases checked\n");

  check_allocator();
  SAY("PRINTF S3: allocator checked\n");

  /* The stdout path, which snprintf does NOT cover: FILE buffering,
     __stdout_write, and SYS_writev. Nothing in the domain can verify this
     arrived -- the run script greps the host's console for this exact line,
     which is also what proves __stdio_exit flushed on the way out. */
  TRACE("before printf to stdout");
  printf("PRINTF STDOUT: %d %s %.2f <end>\n", 42, "ok", 1.5);
  TRACE("printf returned");

  if (failures) {
    SAY("PRINTF FAILED\n");
    return 1;
  }
  SAY("__CAPSTONE_PRINTF_PROBE_PASSED__\n");
  return 0;
}
