#!/usr/bin/env bash
# CHECK (not just a runner) for the Lua-cURLv3 easy<->multi dangling-pointer
# UAF, by DIFFERENTIAL:
#   VULNERABLE b2e9474 -> ASan heap-use-after-free, WRITE inside
#                         lcurl_easy_cleanup (src/lceasy.c), on a multi userdata
#                         that Lua's GC already freed (free frame: l_alloc).
#   FIXED      56b4d05 -> clean run (prints NO-UAF), no ASan report.
# PASS only if BOTH hold. That differential is what makes this a verified
# reproduction of the cross-domain UAF rather than an incidental crash.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd -- "$HERE/.."; pwd)
W="$HERE/.work"
LUA54=$(cd -- "$LC/_toolchain/.work/lua54" 2>/dev/null && pwd || true)
ASAN=$(cc -print-file-name=libasan.so)

[ -x "$LUA54/lua-shared" ]        || { echo "run ./build.sh first (no shared lua)"  >&2; exit 2; }
[ -f "$W/vuln/lcurl.so" ]         || { echo "run ./build.sh first (no vuln lcurl.so)"  >&2; exit 2; }
[ -f "$W/fixed/lcurl.so" ]        || { echo "run ./build.sh first (no fixed lcurl.so)" >&2; exit 2; }

run_trigger() { # $1 = dir holding lcurl.so
  LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUA54" LUA_CPATH="$1/?.so;;" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    "$LUA54/lua-shared" "$HERE/trigger.lua" 2>&1
}

# Outputs go to FILES (grepped below): `... | grep -qa` under pipefail can
# nondeterministically fail on an early match (SIGPIPE); grep on a file cannot.
VOUT="$W/vuln.out"; FOUT="$W/fixed.out"
echo "== vulnerable (b2e9474) =="; run_trigger "$W/vuln"  >"$VOUT" 2>&1; head -6 "$VOUT"
echo "== fixed (56b4d05) =="     ; run_trigger "$W/fixed" >"$FOUT" 2>&1; tail -3 "$FOUT"

# Vulnerable: heap-use-after-free, USE inside lcurl_easy_cleanup (src/lceasy.c),
# and the freed block was released by Lua's GC allocator (l_alloc) -- i.e. it is
# the collected *multi userdata*, proving the cross-object coupling.
vuln_uaf=0
if grep -qa 'heap-use-after-free' "$VOUT" \
   && grep -qa 'lcurl_easy_cleanup' "$VOUT" \
   && grep -qa 'src/lceasy.c' "$VOUT" \
   && grep -qa 'l_alloc' "$VOUT"; then vuln_uaf=1; fi

# Fixed: no ASan report AND the trigger ran to completion.
fix_clean=0
if grep -qa 'NO-UAF' "$FOUT" && ! grep -qa 'AddressSanitizer' "$FOUT"; then fix_clean=1; fi

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: Lua-cURLv3 easy<->multi dangling-pointer UAF reproduced"
  echo "      (heap-use-after-free in lcurl_easy_cleanup on a GC'd multi userdata @ b2e9474; clean @ 56b4d05)"
  exit 0
fi
echo "FAIL: differential not satisfied (vuln_uaf=$vuln_uaf fix_clean=$fix_clean)" >&2
exit 1
