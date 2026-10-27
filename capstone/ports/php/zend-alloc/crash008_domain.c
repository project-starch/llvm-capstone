/* CRASH-008 as a Capstone domain.
 *
 *   php > echo date("U", 999999999999999);
 *
 * PHP 5.0.0's php_date() sizes its output buffer in one pass and writes it in a
 * second. For the 'U' conversion the size pass reserves a FIXED ten bytes:
 *
 *     case 'U':                            ext/standard/datetime.c:358-360
 *         size += 10;
 *         break;
 *
 * and the buffer is then emalloc(size + 1) == emalloc(11) at datetime.c:424.
 * The write pass formats the timestamp as a long and strcats it:
 *
 *     case 'U':                            ext/standard/datetime.c:438-440
 *         sprintf(tmp_buff, "%ld", (long)the_time);
 *         strcat(Z_STRVAL_P(return_value), tmp_buff);
 *
 * 999999999999999 is fifteen digits, so sixteen bytes including the NUL are
 * written into an eleven-byte buffer: a five-byte heap overflow, and a WRITE.
 *
 * WHY IT IS INVISIBLE ON STOCK PHP. REAL_SIZE(11) is 16, so the block that was
 * really allocated is header + 16 and the sixteen-byte write fits it EXACTLY.
 * The corpus measured this: OK under `asan-stock` AND under `asan-nocache`.
 * Only the variant that flattens REAL_SIZE to (size) reports
 * heap-buffer-overflow WRITE of size 16.
 *
 * WHAT THIS DOMAIN SHOWS. The allocator here is unpatched -- REAL_SIZE still
 * rounds, the cache is still on -- but the capability the caller holds is
 * bounded by the true request. The store runs off the end of that capability
 * and the machine stops it. No sanitizer, no redzone, no allocator surgery.
 *
 * ARMS (build both; a fault alone proves nothing)
 *   fault    default                      -> capability fault, cause 29 (out of bounds)
 *   control  -DZEND_CAP_BOUNDS_REAL_SIZE  -> completes, *res = ZEND_CRASH008_SURVIVED
 *
 * The control is the stock-PHP bound. If it does not complete and report, the
 * harness is broken and the fault arm's fault means nothing -- it is not
 * optional politeness (HOW-TO-RUN-ON-QEMU.md:99-120).
 *
 * -O0 IS REQUIRED. At -O1+ the dead store into a buffer nothing reads can be
 * elided outright, and the access under test is never emitted.
 */
#include "zend_capstone_alloc.h"

#define ZEND_CRASH008_SURVIVED  0xC8U   /* control arm reached the end */
#define ZEND_CRASH008_PRESIZE   0xBADU  /* buffer arithmetic disagreed with PHP */

/* --- freestanding libc, only what datetime.c's 'U' arm touches --- */

static unsigned long d_strlen(const char *s)
{
    unsigned long n = 0;
    while (s[n]) { n++; }
    return n;
}

/* strcat, as datetime.c:440 calls it. Deliberately a plain byte loop: this is
 * the instruction sequence under test, and it must run off the end the same way
 * glibc's would. */
static char *d_strcat(char *dst, const char *src)
{
    char *p = dst + d_strlen(dst);
    while ((*p = *src) != '\0') { p++; src++; }
    return dst;
}

/* sprintf("%ld", v) for a non-negative long -- the one conversion datetime.c
 * uses at :439. Returns the length written.
 *
 * Repeated subtraction against a power-of-ten table, NOT `/` and `%`. A 64-bit
 * divide lowers to __udivdi3/__umoddi3, which a freestanding domain does not
 * link; the alternative is pulling in compiler-rt builtins the way
 * ports/sqlite/build-sqlite-capstone.sh:298-307 has to. Nothing here needs them.
 */
static unsigned long d_fmt_long(char *out, unsigned long v)
{
    static const unsigned long pow10[20] = {
        1UL, 10UL, 100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL,
        100000000UL, 1000000000UL, 10000000000UL, 100000000000UL,
        1000000000000UL, 10000000000000UL, 100000000000000UL,
        1000000000000000UL, 10000000000000000UL, 100000000000000000UL,
        1000000000000000000UL, 10000000000000000000UL
    };
    unsigned long n = 0;
    int i = 19;
    while (i > 0 && pow10[i] > v) { i--; }      /* highest place present */
    for (; i >= 0; i--) {
        unsigned digit = 0;
        while (v >= pow10[i]) { v -= pow10[i]; digit++; }
        out[n++] = (char)('0' + digit);
    }
    out[n] = '\0';
    return n;
}

/* --- php_date(), reduced to the 'U' conversion --- */

static unsigned php_date_U(unsigned long the_time)
{
    /* size pass: datetime.c:358-360 */
    unsigned long size = 0;
    size += 10;

    /* datetime.c:424 -- emalloc(size + 1) == emalloc(11) */
    char *out = (char *) _emalloc(size + 1);
    if (!out) { return ZEND_CRASH008_PRESIZE; }
    out[0] = '\0';

    /* write pass: datetime.c:438-440 */
    char tmp_buff[32];
    unsigned long digits = d_fmt_long(tmp_buff, the_time);

    /* The defect is only interesting if the formatted value really is longer
     * than the buffer. Assert the premise rather than assume it: a silent
     * change here would turn the fault arm into a no-op that "passes". */
    if (digits + 1 <= size + 1) { return ZEND_CRASH008_PRESIZE; }

    d_strcat(out, tmp_buff);   /* <-- the five-byte overflow, and the fault */

    /* Only the control arm gets here. Read a byte back so the write cannot be
     * dead-code eliminated even if someone builds this at -O1 by accident. */
    volatile char sink = out[0];
    (void)sink;

    _efree(out);
    return ZEND_CRASH008_SURVIVED;
}

void domain_main(unsigned *res, unsigned func)
{
    (void)func;
    *res = php_date_U(999999999999999UL);
}
