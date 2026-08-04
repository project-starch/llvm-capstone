#!/usr/bin/env bash
# CHECK sol2 #1373 by DIFFERENTIAL: the vuln access (read an interior wrapper
# after the parent userdata is collected) must ASan-UAF; the control (keep the
# parent referenced) must be clean. PASS only if both.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
[ -x "$W/harness" ] || { echo "run ./build.sh first" >&2; exit 2; }
LUADIR="$LC/_toolchain/.work/lua54"; [ -d "$LUADIR" ] && LP="LD_LIBRARY_PATH=$LUADIR" || LP=""

run(){ env $LP ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$W/harness" "$1" 2>&1; }
echo "== vuln =="; V=$(run vuln); echo "$V" | grep -aE 'AddressSanitizer|DONE' | head -2
echo "== control =="; C=$(run control); echo "$C" | grep -aE 'AddressSanitizer|DONE' | head -2

vuln_uaf=0; { echo "$V" | grep -qa 'heap-use-after-free' && echo "$V" | grep -qa 'sol::stack'; } && vuln_uaf=1
ctrl_clean=0; { echo "$C" | grep -qa 'DONE' && ! echo "$C" | grep -qa 'AddressSanitizer'; } && ctrl_clean=1
echo "--- verdict: vuln_uaf=$vuln_uaf ctrl_clean=$ctrl_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$ctrl_clean" = 1 ]; then echo "PASS: sol2 #1373 reproduced"; exit 0; fi
echo "FAIL" >&2; exit 1
