/* Freestanding <inttypes.h> for the Capstone domain.
 *
 * 104 of jerry-core's 200 files fail to compile without this one header -- it is
 * the single cheapest thing standing between JerryScript and this target. clang
 * ships its own <inttypes.h>, but that one #include_next's the host's, which
 * -nostdlibinc removes.
 *
 * Only the PRI* macros jerry-core actually uses are defined. A format macro that
 * is missing is a compile error naming the macro, which is a good failure; one
 * that is defined WRONG prints silent nonsense, which is not. So this stays
 * demand-driven rather than complete.
 */
#ifndef CAPSTONE_ADAPTED_INTTYPES_H
#define CAPSTONE_ADAPTED_INTTYPES_H

#include <stdint.h>

/* LP64: int32_t is int, int64_t is long, intptr_t is long. */
#define PRId8   "d"
#define PRIu8   "u"
#define PRIx8   "x"
#define PRId16  "d"
#define PRIu16  "u"
#define PRIx16  "x"
#define PRId32  "d"
#define PRIi32  "i"
#define PRIu32  "u"
#define PRIx32  "x"
#define PRIX32  "X"
#define PRId64  "ld"
#define PRIi64  "li"
#define PRIu64  "lu"
#define PRIx64  "lx"
#define PRIX64  "lX"
#define PRIdPTR "ld"
#define PRIuPTR "lu"
#define PRIxPTR "lx"

#endif /* CAPSTONE_ADAPTED_INTTYPES_H */
