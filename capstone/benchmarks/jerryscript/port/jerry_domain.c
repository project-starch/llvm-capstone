/* JerryScript inside a Capstone pure-capability domain.
 *
 * The counterpart of ../../micropython/port/mpy_domain.c, and deliberately the same
 * shape: a domain_main that initialises the runtime, runs one thing, and returns a
 * marker in *res. Nothing here is JerryScript-specific except the API calls.
 *
 * THE HEAP IS NOT DECLARED HERE, unlike MicroPython's. jerry_global_heap is a static
 * global inside jcontext.c, sized by JERRY_GLOBAL_HEAP_SIZE. Under
 * -capstone-gp-captable its storage is carved from dom_data at entry exactly the same
 * way, so it costs zero image bytes and charges against the stack budget one for one.
 * That it is upstream's global rather than ours changes nothing about the measurement:
 * it is still one region carved in software.
 *
 * 512 KB is the CEILING, not a choice. JERRY_CPOINTER_32_BIT is 0, so an object
 * reference is a 16-bit offset shifted by JMEM_ALIGNMENT_LOG = 3, and 65535 << 3 is
 * 524280 bytes. The direct-pointer alternative needs the heap below 4 GB because
 * ecma_value_t is uint32_t, and this domain's heap sits above it.
 */
#include <stdint.h>
#include <string.h>
/* The amalgamation bundles every public header into one, so this is the whole API. */
#include "jerryscript.h"
#include "capstone_setjmp.h"

/* ---- the embedder interface, minimal. Every one of these is required to link;
   the ones with no meaning in a domain say so rather than pretending. */

void jerry_port_init (void) {}

/* Set when the runtime gives up, so domain_main can return the reason instead of
   the domain simply never coming back. */
static jmp_buf jd_fatal_env;
static volatile int jd_fatal_armed;
static volatile unsigned jd_fatal_code;

void jerry_port_fatal (jerry_fatal_code_t code) {
  /* A domain has no exit(). Spinning here was honest but useless: a wedge says
     "somewhere after the last marker" and costs a whole boot to learn one bit.
     Longjmp back to domain_main instead, so a runtime that gives up RETURNS the
     fatal code. If nothing armed the buffer we are before setjmp and there is
     nowhere to go, so spin as before rather than jump into a stale frame. */
  jd_fatal_code = (unsigned) code;
  if (jd_fatal_armed) {
    longjmp (jd_fatal_env, 1);
  }
  for (;;) {}
}

void jerry_port_sleep (uint32_t t) { (void) t; }

/* No clock. Returning a FIXED value rather than a plausible one, for the same
   reason get_fattime does in the MicroPython port: a wrong-but-moving clock makes
   Date() confidently wrong, a constant one is visibly constant. */
double jerry_port_current_time (void) { return 0.0; }
int32_t jerry_port_local_tza (double unix_ms) { (void) unix_ms; return 0; }

/* Output. Captured into a static buffer the domain returns a hash of; there is no
   console here. Kept bounded so a runaway print cannot walk off the end -- which
   would be measuring this function rather than the allocator. */
#define JD_OUT_MAX 1024
static char jd_out[JD_OUT_MAX];
static unsigned jd_out_len;

static void jd_emit (const char *p, unsigned n) {
  while (n-- && jd_out_len < JD_OUT_MAX) {
    jd_out[jd_out_len++] = *p++;
  }
}

void jerry_port_print_buffer (const jerry_char_t *buf, jerry_size_t size) {
  jd_emit ((const char *) buf, (unsigned) size);
}

void jerry_port_log (const char *message_p) {
  unsigned n = 0;
  while (message_p[n]) {
    n++;
  }
  jd_emit (message_p, n);
}

/* No filesystem and no REPL. These must exist to link; returning NULL is the
   correct answer for "this domain cannot do that", and every caller checks. */
jerry_char_t *jerry_port_line_read (jerry_size_t *out_size_p) { *out_size_p = 0; return 0; }
void jerry_port_line_free (jerry_char_t *b) { (void) b; }
jerry_char_t *jerry_port_source_read (const char *f, jerry_size_t *o) { (void) f; *o = 0; return 0; }
void jerry_port_source_free (jerry_char_t *b) { (void) b; }
jerry_char_t *jerry_port_path_normalize (const jerry_char_t *p, jerry_size_t n) { (void) n; return (jerry_char_t *) p; }
void jerry_port_path_free (jerry_char_t *p) { (void) p; }
jerry_size_t jerry_port_path_base (const jerry_char_t *p) { (void) p; return 0; }

/* ---- the domain entry */

/* FNV-1a over the captured output. The retval is 32 bits and the output is not, so
   the domain returns a hash and the expected value is computed from a HOST run of the
   same script BEFORE the domain runs -- a prediction that holds, not a number read
   afterwards and declared to agree. Same convention as the MicroPython corpus. */
static unsigned jd_hash (void) {
  unsigned h = 2166136261u;
  for (unsigned i = 0; i < jd_out_len; i++) {
    h ^= (unsigned char) jd_out[i];
    h *= 16777619u;
  }
  return h;
}

static const char jd_script[] = "1+1";

/* JD_STAGE: how far this build goes before returning. EVERY stage returns a
   marker, so a run always yields a result rather than a wedge.
 *
 *   0  return at once                     -- is the domain entered at all?
 *   1  jerry_init + jerry_cleanup
 *   2  ... + eval of a constant
 *   3  ... + eval of "1+1"  (the default; identical to the unstaged build)
 *
 * Kept OUT of the production path deliberately: at the default the code below is
 * the same code it was, so a bisection is about the build that matters. Marker
 * layout stays 0x1E......, with the stage in bits 20..23 and 5 in the low nibble
 * for "runtime gave up", carrying the fatal code in bits 8..15.
 */
#ifndef JD_STAGE
#define JD_STAGE 3
#endif

/* JD_TESTS: run a slice of JerryScript's OWN suite instead of one expression.
 * The batch comes from gen-test-batch.py, which also emits JT_EXPECT_THROW so the
 * tests/jerry/fail/ group is scored the right way round rather than counted as
 * 107 failures.
 *
 * A test that loops forever takes the batch with it and nothing comes back -- that
 * is inherent, and the answer is the same as everywhere else here: run slices and
 * bisect the one that does not return. What must NOT happen is a batch that
 * quietly reports on fewer tests than it was given, so JT_N is what the generator
 * wrote and the loop runs all of it.
 */
#ifdef JD_TESTS
#include "jt_batch.inc"

static unsigned jt_run (void) {
  unsigned pass = 0, fail = 0, first_fail = 0xFF;
  for (unsigned i = 0; i < JT_N; i++) {
    unsigned n = 0;
    while (jt_src[i][n]) n++;
    jerry_value_t v = jerry_eval ((const jerry_char_t *) jt_src[i], n,
                                  JERRY_PARSE_NO_OPTS);
    int threw = jerry_value_is_exception (v) ? 1 : 0;
    jerry_value_free (v);
    if (threw == JT_EXPECT_THROW) {
      pass++;
    } else {
      if (first_fail == 0xFF) first_fail = i;
      fail++;
    }
  }
  /* 9 in the stage nibble marks "test batch". */
  return 0x1E900000u | ((first_fail & 0xFFu) << 16) | ((pass & 0xFFu) << 8)
         | (fail & 0xFFu);
}
#endif

/* Where the result goes, kept where a longjmp cannot lose it.
 *
 * `res` is an ordinary parameter, and capstone_setjmp saves only ra, s0-s11 and sp.
 * If the compiler keeps res in a caller-saved register or a spill slot, its value
 * after longjmp is indeterminate by the C standard and by this implementation --
 * so the "the runtime gave up" marker would be written through a pointer that is
 * whatever survived. A file-scope volatile is not subject to either. */
static unsigned *volatile jd_res;

void domain_main (unsigned *res, unsigned func) {
  (void) func;
  jd_out_len = 0;
  jd_res = res;

#ifdef JD_TESTS
  if (setjmp (jd_fatal_env)) {
    /* The runtime gave up mid-batch. Report that, with the fatal code, rather
       than letting the domain hang and cost a boot to learn one bit. */
    *jd_res = 0x1EA00000u | ((jd_fatal_code & 0xFFu) << 8) | 5u;
    return;
  }
  jd_fatal_armed = 1;
  jerry_init (JERRY_INIT_EMPTY);
  *res = jt_run ();
  jerry_cleanup ();
  return;
#endif

  if (setjmp (jd_fatal_env)) {
    *jd_res = 0x1E000005u | (JD_STAGE << 20) | ((jd_fatal_code & 0xFFu) << 8);
    return;
  }
  jd_fatal_armed = 1;

#if JD_STAGE == 0
  *res = 0x1E000000u | (JD_STAGE << 20) | 6u;
  return;
#else
  jerry_init (JERRY_INIT_EMPTY);
#if JD_STAGE == 1
  jerry_cleanup ();
  *res = 0x1E000000u | (JD_STAGE << 20) | 6u;
  return;
#endif
#if JD_STAGE == 2
  {
    jerry_value_t c = jerry_eval ((const jerry_char_t *) "7", 1, JERRY_PARSE_NO_OPTS);
    unsigned t = jerry_value_is_number (c) ? (unsigned) (int) jerry_value_as_number (c) : 0xEEu;
    jerry_value_free (c);
    jerry_cleanup ();
    *res = 0x1E000000u | (JD_STAGE << 20) | ((t & 0xFFu) << 8) | 6u;
    return;
  }
#endif
#endif

#if JD_STAGE >= 3
  jerry_value_t v = jerry_eval ((const jerry_char_t *) jd_script,
                                sizeof (jd_script) - 1,
                                JERRY_PARSE_NO_OPTS);

  unsigned tag;
  if (jerry_value_is_exception (v)) {
    tag = 2;
  } else if (jerry_value_is_number (v)) {
    tag = 1;
  } else {
    tag = 3;
  }
  double d = jerry_value_is_number (v) ? jerry_value_as_number (v) : 0.0;

  jerry_value_free (v);
  jerry_cleanup ();

  /* 0x1E marks "jerry domain". Low byte is the tag, next byte the integer value of
     the result truncated -- enough to tell 1+1 from a runtime that returned
     something else, without needing the output channel to work. */
  *res = 0x1E000000u | (((unsigned) (int) d & 0xFFu) << 8) | tag;
  if (jd_out_len) {
    *res ^= jd_hash () & 0xFFFF0000u;
  }
#endif /* JD_STAGE >= 3 */
}
