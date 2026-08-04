#!/usr/bin/env bash
# CHECK the LMDB value-after-txn CDP UAF by the read-order differential:
#   vuln    (read AFTER txn commit)  -> ASan heap-use-after-free on the freed page
#   control (read BEFORE txn end)    -> clean, correct value (read ok=true)
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
[ -f "$W/minilmdb.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
run(){ env LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$W/?.so;;" \
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$LUADIR/lua-shared" "$HERE/trigger.lua" "$1" 2>&1; }

V=$(run vuln); C=$(run control)
echo "--- vuln (read after commit) ---";    echo "$V" | grep -aE 'heap-use-after-free|SUMMARY|read ok' | head -3
echo "--- control (read before end) ---";   echo "$C" | grep -aE 'heap-use-after-free|read ok'        | head -3

vok=0; echo "$V" | grep -qa 'heap-use-after-free' && vok=1
cok=0; echo "$C" | grep -qa 'read ok=true'        && cok=1
# control must be clean: no ASan error at all
cclean=1; echo "$C" | grep -qa 'AddressSanitizer' && cclean=0
echo "--- verdict: vuln_uaf=$vok control_ok=$cok control_clean=$cclean ---"
{ [ "$vok" = 1 ] && [ "$cok" = 1 ] && [ "$cclean" = 1 ]; } && { echo "PASS: LMDB value-after-txn UAF reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
