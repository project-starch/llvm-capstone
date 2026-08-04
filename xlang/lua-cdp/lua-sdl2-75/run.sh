#!/usr/bin/env bash
# CHECK lua-SDL2 #75 by DIFFERENTIAL. The hit-test callback builds a SECOND
# Window userdata over the SAME native SDL_Window; on the vulnerable tree that
# duplicate has mustdelete=1, so collecting it runs SDL_DestroyWindow on a
# window the surviving handle + event loop still use.
#
# Vulnerable (96491c0^): after collectgarbage(), win:getID() faults on the freed
#   window -> SDL "Invalid window" (marker WINDOW-DESTROYED-WHILE-LIVE), and the
#   raw read-of-freed-SDL_Window is caught under valgrind.
# Fixed (96491c0, PR #77, mustdelete=0): the window survives GC -> WINDOW-ALIVE,
#   valgrind-clean.
# Both builds must actually fire the hit-test (HITCOUNT>=1) or the control is
# meaningless. PASS only if both halves hold. Needs xvfb + xdotool (headless X).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
TRIG="$HERE/trigger.lua"
[ -f "$W/build-vuln/SDL.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
command -v xvfb-run >/dev/null && command -v xdotool >/dev/null || { echo "need xvfb-run + xdotool" >&2; exit 2; }

run(){ # $1=module dir  $2=optional preload/valgrind mode
  local pre=(); [ "${2:-}" = asan ] && pre=(env LD_PRELOAD="$ASAN")
  xvfb-run -a --server-args="-screen 0 1024x768x24" "${pre[@]}" \
    env LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    "$LUADIR/lua-shared" "$TRIG" 2>&1; }

vg(){ # $1=module dir -> valgrind stderr+stdout (non-ASan build)
  timeout 300 xvfb-run -a --server-args="-screen 0 1024x768x24" \
    env LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" \
    valgrind --error-limit=no --num-callers=25 \
    "$LUADIR/lua-shared" "$TRIG" 2>&1; }

echo "== ASan differential =="
V=$(run "$W/build-vuln" asan);  echo "vuln:  $(echo "$V"|grep -aE 'HITCOUNT|AFTER|WINDOW-'|tr '\n' ' ')"
F=$(run "$W/build-fixed" asan); echo "fixed: $(echo "$F"|grep -aE 'HITCOUNT|AFTER|WINDOW-'|tr '\n' ' ')"

vok=0; { echo "$V"|grep -qa 'HITCOUNT [1-9]' && echo "$V"|grep -qa 'WINDOW-DESTROYED-WHILE-LIVE' \
        && echo "$V"|grep -qa 'Invalid window'; } && vok=1
fok=0; { echo "$F"|grep -qa 'HITCOUNT [1-9]' && echo "$F"|grep -qa 'WINDOW-ALIVE' \
        && ! echo "$F"|grep -qa 'Invalid window'; } && fok=1
echo "--- asan: vuln_uaf=$vok fixed_clean=$fok ---"

vgok=1
if command -v valgrind >/dev/null && [ -f "$W/build-vuln-noasan/SDL.so" ]; then
  echo "== valgrind corroboration (raw read-of-freed SDL_Window) =="
  VV=$(vg "$W/build-vuln-noasan"); FF=$(vg "$W/build-fixed-noasan")
  echo "vuln:  $(echo "$VV"|grep -aE 'Invalid read|block of size 240 free'|head -2|tr '\n' ' ')"
  echo "fixed: $(echo "$FF"|grep -aE 'ERROR SUMMARY'|tail -1)"
  vgok=0
  { echo "$VV"|grep -qaE 'Invalid read' && echo "$VV"|grep -qa "block of size 240 free" \
    && echo "$FF"|grep -qa 'ERROR SUMMARY: 0 errors'; } && vgok=1
  echo "--- valgrind: vuln_uaf=$([ $vgok = 1 ] && echo 1 || echo 0) ---"
else
  echo "== valgrind not available: skipping corroboration (ASan differential stands) =="
fi

if [ "$vok" = 1 ] && [ "$fok" = 1 ] && [ "$vgok" = 1 ]; then
  echo "PASS: lua-SDL2 #75 reproduced (live SDL_Window destroyed by GC of the hit-test's duplicate userdata on the vulnerable tree; clean on the fix)"
  exit 0
fi
echo "FAIL: differential not satisfied" >&2; exit 1
