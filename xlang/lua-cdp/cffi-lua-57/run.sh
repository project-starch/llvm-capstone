#!/usr/bin/env bash
# CHECK cffi-lua #57 by DIFFERENTIAL: the vulnerable commit must ASan-UAF at
# cb_set; the fixed commit must NOT (a clean "bad callback" error). PASS only if
# both hold. This is what makes it a verified reproduction, not a lucky crash.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"
VULN=d295f029a72fa2544eefe0bdbb40d2af8f4f18dc
FIX=ced2cba79
ASAN=$(cc -print-file-name=libasan.so)
[ -d "$W/lua54" ] || { echo "run ./build.sh first" >&2; exit 2; }

[ -d "$W/cffi-lua" ] || git clone https://github.com/q66/cffi-lua "$W/cffi-lua"

build_at() { # $1 = commit -> builds ASan cffi.so
  git -C "$W/cffi-lua" checkout -q "$1"
  rm -rf "$W/cffi-lua/build"
  PKG_CONFIG_PATH="$W/pc" LD_LIBRARY_PATH="$W/lua54" meson setup "$W/cffi-lua/build" "$W/cffi-lua" \
    -Dlua_version=5.4 -Dtests=false -Dbuildtype=debug -Db_sanitize=address >"$W/meson.$1.log" 2>&1 \
    || { echo "meson setup failed for $1 (see $W/meson.$1.log)" >&2; return 1; }
  LD_LIBRARY_PATH="$W/lua54" meson compile -C "$W/cffi-lua/build" >>"$W/meson.$1.log" 2>&1 \
    || { echo "meson compile failed for $1" >&2; return 1; }
  [ -f "$W/cffi-lua/build/cffi.so" ] || { echo "no cffi.so built for $1" >&2; return 1; }
}

run_trigger() { # -> stdout+stderr of the trigger under ASan
  LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$W/lua54" LUA_CPATH="$W/cffi-lua/build/?.so;;" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    "$W/lua54/lua-shared" "$HERE/trigger.lua" 2>&1
}

echo "== vulnerable ($VULN) =="; build_at "$VULN"; V=$(run_trigger); echo "$V" | head -3
echo "== fixed ($FIX) =="; build_at "$FIX"; F=$(run_trigger); echo "$F" | head -3
git -C "$W/cffi-lua" checkout -q "$VULN"   # leave the case on the vulnerable tree

vuln_uaf=0; echo "$V" | grep -qaE 'heap-use-after-free.*cb_set|cb_set.*ffilib.cc:281' && vuln_uaf=1
echo "$V" | grep -qa 'heap-use-after-free' && echo "$V" | grep -qa 'ffilib.cc:281' && vuln_uaf=1
fix_clean=0; { echo "$F" | grep -qa 'bad callback' && ! echo "$F" | grep -qa 'AddressSanitizer'; } && fix_clean=1

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: cffi-lua #57 reproduced (UAF on $VULN, clean on $FIX)"; exit 0
fi
echo "FAIL: differential not satisfied" >&2; exit 1
