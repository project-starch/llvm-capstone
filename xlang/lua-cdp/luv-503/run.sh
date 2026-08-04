#!/usr/bin/env bash
# CHECK luv #503 by differential under valgrind: the vuln tree reads the freed
# coroutine lua_State (ctx->L) inside a luv close/gc callback (lua_settop /
# luaL_unref via luv_close_cb) -> "Invalid read"; the fix (store the main thread
# in ctx->L) runs clean (DONE, no valgrind errors).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"
[ -f "$W/vuln/luv.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
run(){ LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
  valgrind --error-exitcode=99 -q --num-callers=12 "$LUADIR/lua-shared" "$HERE/trigger.lua" 2>&1; }
V=$(run "$W/vuln"); F=$(run "$W/fixed")
# here-strings (not echo|grep): grep -q exits early and, on a >64KB pipe, SIGPIPEs
# echo -> pipefail turns the whole clause into a flaky false negative.
echo "vuln:  $(grep -aE 'Invalid read' <<<"$V"|head -1) | $(grep -aE 'luv_close_cb|luv_gc_cb' <<<"$V"|head -1)"
echo "fixed: $(grep -a DONE <<<"$F"|head -1)"
# vuln: an Invalid read of the freed lua_State, in a luv close/gc cb, via a Lua stack op
vok=0; { grep -qaE 'Invalid read' <<<"$V" && grep -qaE 'luv_close_cb|luv_gc_cb' <<<"$V" \
        && grep -qaE 'lua_settop|luaL_unref|lua_rawgeti' <<<"$V" && grep -qa "free'd" <<<"$V"; } && vok=1
# fixed: reaches DONE with zero valgrind errors
fok=0; { grep -qa DONE <<<"$F" && ! grep -qaE 'Invalid read|Invalid write' <<<"$F"; } && fok=1
echo "--- verdict: vuln_uaf=$vok fixed_ok=$fok ---"
{ [ "$vok" = 1 ] && [ "$fok" = 1 ]; } && { echo "PASS: luv #503 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
