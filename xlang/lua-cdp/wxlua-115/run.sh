#!/usr/bin/env bash
# CHECK wxLua #115 by DIFFERENTIAL: the vulnerable commit must ASan
# heap-use-after-free (double-delete) at wxLua_wxMenu_delete_function during the
# submenu userdata's __gc; the fixed commit must NOT (reaches "NO-CRASH").
# PASS only if both hold. Headless via xvfb-run (wxMenu needs a wxApp).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"
[ -x "$W/vuln/wxLua" ] && [ -x "$W/fix/wxLua" ] || { echo "run ./build.sh first" >&2; exit 2; }
command -v xvfb-run >/dev/null || { echo "xvfb-run not found" >&2; exit 2; }

run_trigger() { # $1 = vuln|fix -> stdout+stderr of the trigger under ASan, headless
  local d="$W/$1"; local lua_inc; lua_inc=$(cat "$d/.lua_inc")
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    LD_LIBRARY_PATH="$d:$lua_inc" \
    xvfb-run -a "$d/wxLua" "$HERE/trigger.lua" 2>&1
}

echo "== vulnerable =="; V=$(run_trigger vuln); echo "$V" | grep -aE 'AddressSanitizer|delete_function|NO-CRASH' | head -3
echo "== fixed ==";      F=$(run_trigger fix);  echo "$F" | grep -aE 'AddressSanitizer|NO-CRASH' | head -3

# Vulnerable signature: heap-use-after-free whose use site is
# wxLua_wxMenu_delete_function at wxcore_menutool.cpp:1387, reached through the
# wxLua GC path (deletegcobject / __gc). Line 1387 pins the exact `delete o`.
vuln_uaf=0
if echo "$V" | grep -qa 'heap-use-after-free' \
   && echo "$V" | grep -qa 'wxLua_wxMenu_delete_function' \
   && echo "$V" | grep -qa 'wxcore_menutool.cpp:1387' \
   && echo "$V" | grep -qaE 'wxluaO_deletegcobject|wxlua_wxLuaBindClass__gc|GCTM'; then
  vuln_uaf=1
fi

# Fixed signature: clean run (reaches the print), no ASan report.
fix_clean=0
{ echo "$F" | grep -qa 'NO-CRASH: reached end' && ! echo "$F" | grep -qa 'AddressSanitizer'; } && fix_clean=1

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: wxLua #115 reproduced (double-delete UAF on vuln, clean on fix)"; exit 0
fi
echo "FAIL: differential not satisfied" >&2; exit 1
