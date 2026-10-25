#include "corpus.h"

/* NEGATIVE CONTROL, under -DPGCLIENT_NEGATIVE_CONTROL.
 *
 * SCHEMA rule 5. A fault is this case's result only if it depends on the
 * defect, and an arm that would fault on the same shape just below the
 * threshold is reporting something else. The control keeps the allocation,
 * the call and the function under test, and moves ONE value to the safe side
 * of the boundary the defect crosses. It must complete.
 */

PGCLIENT_CASE(3) {
/* Overread in SASLprep's UTF-8 validation. Fixed by 5d61bdd114; live at the
 * 17.5 pin.
 *
 *   src/common/saslprep.c:1013-1022
 *
 *     while (*p)
 *     {
 *         l = pg_utf_mblen(p);           -- reads the LEAD byte only, and
 *                                          returns the length it DECLARES
 *         if (!pg_utf8_islegal(p, l))    -- then reads l bytes
 *             return -1;
 *         p += l;
 *     }
 *
 * When the terminating NUL falls inside the declared sequence, islegal reads
 * past it and past the end of the allocation. `while (*p)` cannot help: it is
 * evaluated at the lead byte, which is non-NUL in exactly the failing case.
 *
 * WHY THIS IS NON-NESTED, and why this case earns its place. saslprep.c is
 * compiled twice: ALLOC is palloc under the backend and, at saslprep.c:48,
 * malloc under FRONTEND. libpq takes the FRONTEND copy, so the same source
 * line is a nested defect in one binary and a non-nested one in the other,
 * with nothing else changed. */

  /* The password an application handed to PQconnectdb, ending in a lead byte
   * that declares a three-byte sequence it does not have. Sized exactly: the
   * allocation ends at the NUL, so the overread leaves it. */
#ifdef PGCLIENT_NEGATIVE_CONTROL
  /* The same length and the same allocation, with the truncated lead byte
   * replaced by an ASCII character. pg_utf_mblen then returns 1 where it
   * returned 3, and the walk stops at the terminator inside the buffer. */
  static const char truncated[] = "pwx";
#else
  static const char truncated[] = "pw\xE2";
#endif
  size_t  n = sizeof(truncated);              /* 4: 'p','w',0xE2,'\0' */
  char   *password = pgclient_malloc(n);

  pgclient_memcpy(password, truncated, n);

  pgclient_expect_fault_in((const void *) &pg_utf8_string_len, "pg_utf8_string_len");
  /* The read is in pg_utf8_islegal, which is static upstream and has no
   * address to name; it is called from here, so the RETURN ADDRESS is
   * what should land in pg_utf8_string_len. */
  pgclient_mark();                    /* the defect's own line is the next one */
  (void) pg_utf8_string_len(password);

  /* Unreached on a mechanism that bounds the read: pg_utf_mblen returned 3 for
   * the 0xE2 at offset 2, so islegal read offsets 2..4 of a 4-byte object. */
  pgclient_note_overread(password, n, n + 1);
  pgclient_free(password);
}
