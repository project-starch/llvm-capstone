#!/usr/bin/env bash
# CHECK LuaBridge #319 by the destructor-sentinel differential: the chained
# call (vuln) reads a destroyed temporary -> getI() returns -1; the non-chained
# control owns a valid copy -> 5. (ASan is unreliable here: Lua reuses the freed
# userdata block, so the -1 sentinel is the deterministic proof.)
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
[ -x "$W/harness" ] || { echo "run ./build.sh first" >&2; exit 2; }
LUADIR="$LC/_toolchain/.work/lua54"; [ -d "$LUADIR" ] && LP="LD_LIBRARY_PATH=$LUADIR" || LP=""
V=$(env $LP ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$W/harness" 2>&1)
C=$(env $LP ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$W/harness" control 2>&1)
echo "vuln:    $(echo "$V" | grep -a got)"
echo "control: $(echo "$C" | grep -a got)"
vuln_uaf=0; echo "$V" | grep -qa 'got.*-1' && vuln_uaf=1
ctrl_ok=0;  echo "$C" | grep -qa 'got.*5'  && ctrl_ok=1
echo "--- verdict: vuln_uaf(sentinel -1)=$vuln_uaf ctrl_ok(5)=$ctrl_ok ---"
{ [ "$vuln_uaf" = 1 ] && [ "$ctrl_ok" = 1 ]; } && { echo "PASS: LuaBridge #319 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
