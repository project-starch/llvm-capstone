#!/usr/bin/env bash
# CHECK ldbus #20 by differential under a private session bus: vuln reads a
# reused message -> arg_type=nil; fixed keeps the reply -> arg_type=a.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
[ -f "$W/vuln/ldbus.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
run(){ dbus-run-session -- env LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$LUADIR/lua-shared" "$HERE/trigger.lua" 2>&1; }
V=$(run "$W/vuln"); F=$(run "$W/fixed")
echo "vuln:  $(echo "$V"|grep -a arg_type)"; echo "fixed: $(echo "$F"|grep -a arg_type)"
vok=0; echo "$V"|grep -qa 'arg_type=nil' && vok=1
fok=0; echo "$F"|grep -qa 'arg_type=a'   && fok=1
echo "--- verdict: vuln_uaf(nil)=$vok fixed_ok(a)=$fok ---"
{ [ "$vok" = 1 ] && [ "$fok" = 1 ]; } && { echo "PASS: ldbus #20 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
