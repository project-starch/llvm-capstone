/* Single globals-owning module for the gp-captable Lua domain.
 *
 * The gp-captable ABI's init descriptor (.capstone_gp_initdesc, emitted by
 * CapstoneAsmPrinter::emitGpCaptableInitDesc) carries a per-MODULE header
 * {built_flag,count,gp_slot} followed by that module's records. The entry glue
 * (start-gp-captable-interp.S) reads ONE header's `count` at blob offset 0, so
 * only ONE module in the whole domain may own gp-captable globals -- a second
 * module's header would be read as records and the table would be built wrong.
 * SQLite solves this by amalgamating everything-with-globals into one TU; this
 * does the same for Lua.
 *
 * Everything here owns gp-captable globals (audited): the allocator's arena/slot
 * state, the softfloat libm's constant tables, the libc's tables, Lua's static
 * capability tables (base_funcs[] etc.), and the harness's string constants. The
 * only objects linked SEPARATELY are provably globals-free: beebs_string
 * (BEEBS_STRING_LINEAR_SAFE), the clean compiler-rt double builtins, the
 * globals-free floatdidf, and the hand-written setjmp.S.
 *
 * Support files come first so lua_amalg.c's `#define LUAI_FUNC static` and
 * friends do not leak into them. The freestanding libc surface
 * (capstone_lua_libc.h) is force-included ahead of this file, so every TU here
 * sees the same declarations regardless of order.
 */

/* allocator: malloc/free/realloc/xlang_arena_init + the revoking rof state */
#include "revoke_arena_domain.c"

/* libm: floor/pow/exp/fmod/... (softfloat, with constant tables) */
#include "beebs_softfloat_libm.c"

/* __floatdidf (int64->double): at -O0 it materializes two IEEE constants as
 * .rodata locals, so it owns gp-captable globals and must live in this one module.
 * SQLite keeps it a SEPARATE object only because SQLITE_OMIT_FLOATING_POINT does
 * `#define double sqlite_int64`, which poisons compiler-rt inside the amalgamation;
 * this domain uses real doubles, so folding it in is both legal and required. */
#include "capstone_floatdidf_noglobals.c"

/* libc gaps: snprintf/strtod/ctype/locale + hosted stubs */
#include "lua_libc.c"

/* the reference interpreter: Lua 5.4.7 core + base + coroutine libs */
#include "lua_amalg.c"

/* the domain harness: domain_main, run_lua()/run_lua_staged(), the chunk */
#include "lua_domain.c"
