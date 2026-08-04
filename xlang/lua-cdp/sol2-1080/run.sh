#!/usr/bin/env bash
# CHECK sol2 #1080 by DIFFERENTIAL: the vuln path (sol stores a "Foo*" pointer to
# the native object, which is then delete'd) must ASan heap-use-after-free when a
# later Lua->C++ call (test:Print) re-derefs the freed Foo; the control path (sol
# copies into a Lua-owned "Foo" value userdata) must be clean despite the same
# delete. PASS only if both hold.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
[ -x "$W/harness" ] || { echo "run ./build.sh first" >&2; exit 2; }
LUADIR="$LC/_toolchain/.work/lua54"; [ -d "$LUADIR" ] && LP="LD_LIBRARY_PATH=$LUADIR" || LP=""

run(){ env $LP ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$W/harness" "$1" 2>&1; }
echo "== vuln =="; V=$(run vuln); echo "$V" | grep -aE 'AddressSanitizer|DONE|__name=' | head -3
echo "== control =="; C=$(run control); echo "$C" | grep -aE 'AddressSanitizer|DONE|__name=' | head -3

# vuln: a heap-use-after-free whose faulting read is the sol->C++ method dispatch
# on the dangling Foo* userdata (Foo::Print reads this->val of the freed block).
vuln_uaf=0; { echo "$V" | grep -qa 'heap-use-after-free' && echo "$V" | grep -qa 'Foo::Print'; } && vuln_uaf=1
# control: value-copy survives the native delete — completes, no ASan report.
ctrl_clean=0; { echo "$C" | grep -qa 'DONE' && ! echo "$C" | grep -qa 'AddressSanitizer'; } && ctrl_clean=1
echo "--- verdict: vuln_uaf=$vuln_uaf ctrl_clean=$ctrl_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$ctrl_clean" = 1 ]; then echo "PASS: sol2 #1080 reproduced"; exit 0; fi
echo "FAIL" >&2; exit 1
