#!/usr/bin/env bash
# CHECK xmlua #35 by valgrind differential: vuln frees the document before the
# xpath object -> xmlXPathFreeNodeSet reads freed node->type (Invalid read);
# control frees the xpath object first -> clean.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); W="$HERE/.work"
[ -d "$W/xmlua" ] || { echo "run ./build.sh first" >&2; exit 2; }
LP="$W/xmlua/?.lua;$W/xmlua/?/init.lua;$W/luacs/?.lua;$W/luacs/?/init.lua;;"
vg(){ LUA_PATH="$LP" valgrind -q --error-exitcode=99 luajit -joff "$1" 2>&1; }
V=$(vg "$HERE/trigger.lua"); vrc=$?
C=$(vg "$HERE/trigger-control.lua"); crc=$?
echo "vuln:    rc=$vrc $(echo "$V"|grep -a 'Invalid read'|head -1)"
echo "control: rc=$crc $(echo "$C"|grep -a DONE|head -1)"
vok=0; { [ $vrc -eq 99 ] && echo "$V"|grep -qa 'xmlXPathFreeNodeSet'; } && vok=1
cok=0; { [ $crc -eq 0 ] && echo "$C"|grep -qa DONE && ! echo "$C"|grep -qa 'Invalid read'; } && cok=1
echo "--- verdict: vuln_uaf=$vok control_clean=$cok ---"
{ [ "$vok" = 1 ] && [ "$cok" = 1 ]; } && { echo "PASS: xmlua #35 reproduced"; exit 0; }
echo "FAIL" >&2; exit 1
