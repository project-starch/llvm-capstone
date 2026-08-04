#!/usr/bin/env bash
# CHECK luv #503 by differential: vuln reports a heap-use-after-free reading the
# freed coroutine lua_State in luv_close_cb/luv_gc_cb (handle __gc); the fix
# (store the main thread in ctx->L) runs clean (DONE).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
[ -f "$W/vuln/luv.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
run(){ LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$LUADIR/lua-shared" "$HERE/trigger.lua" 2>&1; }
V=$(run "$W/vuln"); F=$(run "$W/fixed")
echo "vuln:  $(echo "$V"|grep -aE 'heap-use-after-free|luv_close_cb|luv_gc_cb'|head -1)"
echo "fixed: $(echo "$F"|grep -a DONE|head -1)"
vok=0; { echo "$V"|grep -qaE 'heap-use-after-free|SEGV' && echo "$V"|grep -qaE 'luv_close_cb|luv_gc_cb'; } && vok=1
fok=0; { echo "$F"|grep -qa DONE && ! echo "$F"|grep -qa 'AddressSanitizer'; } && fok=1
echo "--- verdict: vuln_uaf=$vok fixed_ok=$fok ---"
{ [ "$vok" = 1 ] && [ "$fok" = 1 ]; } && { echo "PASS: luv #503 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
