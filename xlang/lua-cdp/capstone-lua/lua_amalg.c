/* Trimmed single-TU amalgamation of reference Lua 5.4.7 for a Capstone domain.
 *
 * This is onelua.c's exact structure, minus the syscall-heavy libraries
 * (io/os/loadlib/debug/utf8) and the standalone `lua.c` main. Building Lua as
 * ONE translation unit is REQUIRED on Capstone: multiple globals-owning TUs
 * collide on the single gp cap-table (positional indices), so their static
 * capability tables (base_funcs[], string constants) never get tagged by the
 * .capstone_cap_init mechanism -> luaopen_base reads an untagged capability and
 * faults (cause 24). SQLite is amalgamated for exactly this reason; so is Lua.
 * Every OTHER TU linked into the domain (harness, libc, libm) must be
 * cap-global-free, so this stays the sole cap-globals owner.
 *
 * The reference l*.c files are found via -I<lua source> (they are not vendored).
 */
#include "lprefix.h"

#define LUA_CORE
#define LUA_LIB
#define ltable_c
#define lvm_c
#include "luaconf.h"

/* Hide internal symbols so this is a self-contained unit (onelua.c does this). */
#undef LUAI_FUNC
#undef LUAI_DDEC
#undef LUAI_DDEF
#define LUAI_FUNC	static
#define LUAI_DDEC(def)	/* empty */
#define LUAI_DDEF	static

/* core -- used by all */
#include "lzio.c"
#include "lctype.c"
#include "lopcodes.c"
#include "lmem.c"
#include "lundump.c"
#include "ldump.c"
#include "lstate.c"
#include "lgc.c"
#include "llex.c"
#include "lcode.c"
#include "lparser.c"
#include "ldebug.c"
#include "lfunc.c"
#include "lobject.c"
#include "ltm.c"
#include "lstring.c"
#include "ltable.c"
#include "ldo.c"
#include "lvm.c"
#include "lapi.c"

/* auxiliary library + the base/coroutine libraries.
 * Excluded: liolib, loslib, loadlib, ldblib, lutf8lib (syscalls / full stdio),
 * lmathlib/lstrlib/ltablib (extra libm transcendentals not yet provided), and
 * linit.c (it registers io/os) -- the domain registers what it needs. Adding the
 * math/string/table libs later only needs asin/atan2/tan/log2/log10/sinh/cosh/
 * tanh in the libm. */
#include "lauxlib.c"
#ifndef LUA_CORE_ONLY
#include "lbaselib.c"
#include "lcorolib.c"
#endif
