/* Two symbols musl cannot supply yet, stubbed HONESTLY.
 *
 * Neither is missing because a domain lacks the OS service. Both are missing
 * because the musl source file does not compile for this target:
 *
 *   src/stdio/fopen.c     absent from libc-capstone.a
 *   src/stdio/vfprintf.c  absent -- backend assertion in APInt::getSExtValue,
 *                         the 128-bit long-double family (see README)
 *
 * `fprintf.o` IS in the archive and references `vfprintf`, and lauxlib's
 * `luaL_loadfilex` references `fopen`, so both are undefined at link even
 * though this probe calls neither. `--gc-sections` does not help: lld resolves
 * symbols before it collects, so a reference from a section that will be
 * discarded still has to resolve.
 *
 * They therefore FAIL LOUDLY rather than pretending. A silent stub here would
 * make "Lua ran" mean less than it should: file loading and formatted output
 * genuinely do not work in this build, and a run that needed them must say so
 * instead of producing a plausible-looking pass.
 */
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>

extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);

FILE *fopen(const char *path, const char *mode) {
  (void)path;
  (void)mode;
  __capstone_hc_write(1, "LUA STUB: fopen is not available in this build\n", 46);
  errno = ENOSYS;
  return 0;
}

int vfprintf(FILE *stream, const char *format, va_list ap) {
  (void)stream;
  (void)format;
  (void)ap;
  __capstone_hc_write(1, "LUA STUB: vfprintf is not available in this build\n", 49);
  errno = ENOSYS;
  return -1;
}

/* strtod and friends: LOUD stubs, to cut `long double` out of the link.
 *
 * musl's strtod goes through src/internal/floatscan.c, which converts via
 * `long double` and therefore pulls __addtf3, __multf3, __getf2, fabsl,
 * copysignl, fmodl and the rest of the 128-bit family. NONE of those exist:
 * every long-double builtin in compiler-rt fails to compile for this target
 * with the same three backend assertions as musl's own long-double files
 * (measured; see README). `long double` is currently unusable on capstone64,
 * in musl and in compiler-rt alike.
 *
 * Providing strtod here means strtod.o is never pulled from the archive, and
 * floatscan.o with it, and 15 undefined symbols disappear at once.
 *
 * A silent partial implementation would be the wrong trade: Lua's lexer uses
 * strtod for FLOAT literals, so a subtly wrong parser would make a chunk
 * compute a wrong number and still report success. These say so instead. The
 * probe's chunk uses integer literals only, which Lua reads with its own
 * l_str2int, so this path is linked but not taken.
 */
double strtod(const char *s, char **end) {
  (void)s;
  if (end)
    *end = (char *)s;
  __capstone_hc_write(1, "LUA STUB: strtod is not available (long double)\n", 47);
  return 0.0;
}

float strtof(const char *s, char **end) { return (float)strtod(s, end); }
long double strtold(const char *s, char **end) { return strtod(s, end); }
