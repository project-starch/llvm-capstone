/* Phase markers for the CHERI/PHP-allocator rows.
 *
 * WHY PHASES AND NOT JUST AN EXIT STATUS: the exit status alone cannot say
 * WHERE a capability fault happened, and for this experiment that is the whole
 * question. PHP's shutdown_memory_manager() drains the size-class cache and
 * free()s every block, so a block efree'd mid-request is released only at
 * shutdown -- long after the trigger's use site. A fault at shutdown is NOT a
 * catch of the defect: the program already consumed the stale data.
 *
 * So each row prints a marker before and after each phase, UNBUFFERED, and the
 * classifier reads the last marker to survive. Phases:
 *   LAYOUT  - the allocator geometry this build actually has (evidence, not prose)
 *   USE     - the offending access. A fault here is a catch AT THE CONTRACT POINT.
 *   SHUT    - shutdown_memory_manager(). A fault here is a late catch only.
 *   DONE    - the defect survived end to end.
 */
#ifndef ROW_H
#define ROW_H

#include <stdio.h>
#include <stdlib.h>
#include "php_alloc.h"

/* THE OFFENDING ACCESS MUST BE FIXED, AND ONLY THE ALLOCATOR MAY VARY.
 *
 * The spatial rows read/write a byte at a FIXED offset derived from the REAL
 * PHP geometry, never from the geometry of the arm being compiled. Deriving it
 * from the live REAL_SIZE() made the patched arm read offsets [26..25] -- an
 * empty range -- so that arm performed no over-read at all and could not
 * possibly be caught. A control the treatment arm cannot fail is not a control.
 *
 * STOCK_REAL_SIZE is therefore PHP 5.0.0's rounding, hard-coded, so both arms
 * touch the SAME byte: in-bounds slack under the real allocator, past the end
 * of the block once the corpus patches remove the rounding. */
#define STOCK_REAL_SIZE(n) ((((n)) + 7) & ~0x7)

#define ROW_MARK(tag) do { printf("[%s]\n", tag); fflush(stdout); } while (0)

static void row_begin(const char *name)
{
    setvbuf(stdout, (char *) NULL, _IONBF, 0);
    printf("ROWNAME %s\n", name);
    zend_alloc_startup();
    printf("LAYOUT hdr=%zu cache=%d rs(1)=%zu rs(26)=%zu rs(32)=%zu\n",
           php_alloc_header_bytes(), php_alloc_cache_is_on(),
           php_alloc_real_size(1), php_alloc_real_size(26),
           php_alloc_real_size(32));
    fflush(stdout);
}

/* Keep the compiler from eliding a dangling or out-of-bounds access. Rows are
 * built -O0 as well, for the reason the SQLite baseline records, but a sink is
 * cheap insurance and makes the row honest at any -O. */
static volatile unsigned long row_sink;
static void row_consume(const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *) p;
    unsigned long a = 0;
    size_t i;
    for (i = 0; i < n; i++) { a += b[i]; }
    row_sink += a;
}

/* Variadic because a row body may contain a top-level comma (e.g. a single
 * declaration of two variables): macro arguments respect parentheses, not
 * braces, so a fixed second parameter splits such a body. */
#define ROW_MAIN(name, ...)                    \
    int main(void)                             \
    {                                          \
        row_begin(name);                       \
        ROW_MARK("USE-BEGIN");                 \
        { __VA_ARGS__ }                        \
        ROW_MARK("USE-END");                   \
        ROW_MARK("SHUT-BEGIN");                \
        shutdown_memory_manager();             \
        ROW_MARK("SHUT-END");                  \
        ROW_MARK("DONE");                      \
        return 0;                              \
    }

#endif /* ROW_H */
