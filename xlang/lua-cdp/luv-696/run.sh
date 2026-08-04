#!/usr/bin/env bash
# CHECK luv #696 by differential: vuln SEGVs in luv_fs_gc (scandir req __gc
# cleans up a uv_fs_t a worker thread uses); fixed runs clean (DONE).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
[ -f "$W/vuln/luv.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
run(){ LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$LUADIR/lua-shared" "$HERE/trigger.lua" 2>&1; }
V=$(run "$W/vuln"); F=$(run "$W/fixed")
echo "vuln:  $(echo "$V"|grep -aE 'SEGV|heap-use-after-free|luv_fs_gc'|head -1)"
echo "fixed: $(echo "$F"|grep -a DONE|head -1)"
vok=0; { echo "$V"|grep -qaE 'SEGV|heap-use-after-free' && echo "$V"|grep -qa 'luv_fs_gc'; } && vok=1
fok=0; { echo "$F"|grep -qa DONE && ! echo "$F"|grep -qa 'AddressSanitizer'; } && fok=1
echo "--- verdict: vuln_uaf=$vok fixed_ok=$fok ---"
{ [ "$vok" = 1 ] && [ "$fok" = 1 ]; } && { echo "PASS: luv #696 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
