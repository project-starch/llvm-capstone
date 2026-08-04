#!/usr/bin/env bash
# CHECK lgi #122 by valgrind differential (cairo.Region boxed UAF).
#   vuln  + trigger.lua          -> Invalid read in cairo_region_get_extents,
#                                   freed by record_gc (g_boxed_free) via lua_gc.
#   fixed + trigger.lua          -> clean (the metatable-nil guard blocks re-use).
#   vuln  + trigger-control.lua  -> clean (safe access, no finaliser resurrection).
# PASS only if the vuln UAF fires AND both clean runs are clean.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"
[ -f "$W/vuln/lgi/corelgilua51.so" ] || { echo "run ./build.sh first" >&2; exit 2; }

vg(){ # $1 = tree root, $2 = script
  LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$1/?.so;;" LUA_PATH="$1/?.lua;$1/?/init.lua;;" \
    valgrind -q --error-exitcode=99 "$LUADIR/lua-shared" "$2" 2>&1; }

V=$(vg "$W/vuln"  "$HERE/trigger.lua");         vrc=$?
F=$(vg "$W/fixed" "$HERE/trigger.lua");         frc=$?
C=$(vg "$W/vuln"  "$HERE/trigger-control.lua"); crc=$?

echo "vuln:    rc=$vrc $(echo "$V"|grep -a 'Invalid read'|head -1)"
echo "         $(echo "$V"|grep -aE 'cairo_region_get_extents|record_gc'|head -2|tr '\n' ' ')"
echo "fixed:   rc=$frc $(echo "$F"|grep -a DONE|head -1)"
echo "control: rc=$crc $(echo "$C"|grep -a DONE|head -1)"

vok=0; { [ $vrc -eq 99 ] && echo "$V"|grep -qa 'cairo_region_get_extents' && echo "$V"|grep -qa 'record_gc'; } && vok=1
fok=0; { [ $frc -eq 0 ] && echo "$F"|grep -qa DONE && ! echo "$F"|grep -qa 'Invalid read'; } && fok=1
cok=0; { [ $crc -eq 0 ] && echo "$C"|grep -qa DONE && ! echo "$C"|grep -qa 'Invalid read'; } && cok=1
echo "--- verdict: vuln_uaf=$vok fixed_clean=$fok control_clean=$cok ---"
{ [ "$vok" = 1 ] && [ "$fok" = 1 ] && [ "$cok" = 1 ]; } && { echo "PASS: lgi #122 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
