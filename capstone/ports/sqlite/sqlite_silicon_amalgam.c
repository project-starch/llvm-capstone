/* Single translation unit for the SILICON SQLite domain (issue RISK A / stage S4).
 *
 * WHY THIS FILE EXISTS. Under `-capstone-gp-captable` the compiler numbers globals
 * PER MODULE and positionally (`getGpCaptableIndex`), and there is exactly ONE gp
 * cap-table at runtime. Two translation units that each own globals therefore both
 * start at index 0, and both lower to `ldc rd, 0(gp)` -- reading each other's
 * storage. It is silent: wrong data, no fault.
 *
 * WHICH FILES HAVE TO BE HERE, measured rather than assumed (llvm-nm over the
 * existing build's objects, counting defined data symbols):
 *
 *     sqlite3.o     996      the amalgamation, already one TU
 *     domain.o       72      the domain driver
 *     floatdidf.o     2      compiler-rt: __floatdidf.twop32 / .twop52
 *     sqlite_vfs.o    1
 *     sqlite_os.o     1
 *     the other 14    0      start, libc, beebs_string, 11 more builtins
 *
 * A TU with no globals emits no descriptor and no `ldc gp[i]`, so it cannot collide
 * and does not need to be here. That is why this is five files and not nineteen --
 * the original scoping assumed the whole 19-object link had to be merged.
 *
 * `floatdidf` is the awkward one: third-party compiler-rt, and its two globals are
 * function-local `double` constants. It is pulled in LAST so its macros cannot
 * disturb SQLite, and it is the only compiler-rt file here -- the other 11 builtins
 * SQLite references (`__eqdf2`, `__fixdfdi`, `__muldf3`, ...) own no data and stay
 * as separate objects.
 *
 * Note the float builtins cannot simply be omitted: `SQLITE_OMIT_FLOATING_POINT=1`
 * is set and `sqlite3.o` still references nine of them undefined.
 *
 * Order matters: SQLite first (it is the bulk and defines the most macros), then our
 * small TUs, then third-party. Each include is guarded by the build script having
 * already produced/patched the file.
 */

/* 1. The patched SQLite amalgamation (build-sqlite-capstone.sh's sed pass output). */
#include "sqlite3-capstone.c"

/* 2. Our support TUs that own globals. */
#include "capstone_sqlite_vfs.c"
#include "capstone_sqlite_os.c"

/* 3. The domain driver (domain_main and its state). */
#include "sqlite_capstone_domain.c"

/* 4. NOT compiler-rt's floatdidf.c -- it cannot be amalgamated. SQLite's
      SQLITE_OMIT_FLOATING_POINT does `#define double sqlite_int64`, which poisons
      compiler-rt's int_types.h in the same TU (typedef redefinition, 'long' vs
      'long long'). A globals-free replacement is compiled as its OWN object instead:
      capstone_floatdidf_noglobals.c. Having no globals is what lets it stay separate. */
