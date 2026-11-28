#!/bin/bash
# build-gcc-v2.sh (nested lane, 2026-11-26): CFP-g3 v2, gcc's all-abc-bm-ls-tfl-phi port with a fail-stop CRENEW, built
# exactly as build-ports.sh builds gcc-port (B5TL1's own 502.gcc_r tree, its compile.txt and link.txt, llvm-c9, the
# B5TL1 SDK; the 11 port sources of oneop/gcc-arms/all-abc-bm-ls-tfl-phi, each first checked equal to oneop's stock
# source), with one difference: sublet-port.h is gcc-v2/sublet-port.h, whose sublet_renew ends the old lifetime, writes
# SUBLET-RENEW-FAILED to stderr and aborts when CRENEW returns 0. Arms:
#   gcc-port2    the v2 port
#   gcc-port2ff  the v2 port with -DSUBLET_RENEW_FORCE_FAIL on every compile line: every renewal reads as failed
#                without executing CRENEW, so the process must abort at its first renew with the line (the control)
main() {
  set -u
  for v in $(env | sed -n 's/^\(CAPSTONE_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  export CAPSTONE_LLVM_BIN=<hdd>/capstone/llvm-c9/bin
  local X=<hdd>/capstone/nested/b5b2/ports T=<hdd>/capstone/perf/cclever/arms/B5TL1
  local G=<hdd>/capstone/nested/oneop/gcc-arms/all-abc-bm-ls-tfl-phi V=<hdd>/capstone/nested/b5b2/ports/gcc-v2
  local GS=<hdd>/capstone/nested/oneop/spec/tree/SPEC17/benchspec/CPU/502.gcc_r/build/build_base_oneop.0000
  local OC=<hdd>/capstone/llvm-c4/bin/llvm-objcopy SCAN=<hdd>/capstone/nested/b5b2/scan_illegal.py
  local KS="--strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq"
  local PIN="taskset -c 16-27,48-59 nice -n 10"
  local GFILES="alloc-pool.c bitmap.c c-decl.c df-scan.c emit-rtl.c ggc-page.c gimple.c include/obstack.h lists.c obstack.c tree-phinodes.c"
  { echo "compiler $($CAPSTONE_LLVM_BIN/clang --version | head -1)"; sha256sum $CAPSTONE_LLVM_BIN/clang $T/sdk/capstone-cc $V/sublet-port.h $G/sublet-port.h; } > $V/inputs.txt
  for arm in ${ARMS:-gcc-port2 gcc-port2ff}; do
    local D=$V/$arm exe=cpugcc_r defs=""
    [ -e $D ] && { echo "$D exists"; return 2; }
    cp -a $T/spec/502.gcc_r $D && find $D -name '*.o' -type f -delete && rm -f $D/$exe || return 1
    local f; for f in $GFILES; do cmp -s $GS/$f $D/$f || { echo "$arm: B5TL1's $f differs from oneop's stock"; return 1; }; cp $G/$f $D/$f || return 1; done
    cp $V/sublet-port.h $D/ || return 1
    [ $arm = gcc-port2ff ] && defs="-DSUBLET_RENEW_FORCE_FAIL"
    python3 - $D "$defs" <<'PY' || return 1
import sys
d, defs = sys.argv[1:3]
L = open(d + '/compile.txt').read().splitlines()
out = []
for l in L:
    if not l.strip(): continue
    assert l.count(' -c -o ') == 1, l[:120]
    out.append(l.replace(' -c -o ', ' ' + defs + ' -c -o ', 1) if defs else l)
open(d + '/compile-arm.txt', 'w').write('\n'.join(out) + '\n')
print('compile-arm.txt', len(out), 'lines, defs [%s]' % defs)
PY
    ( cd $D && $PIN xargs -a compile-arm.txt -d '\n' -P 11 -n 1 bash -c 'eval "$1" > /dev/null 2>> compile-arm.err || echo "FAIL $1"' _ > fail )
    [ -s $D/fail ] && { echo "$arm COMPILE-FAIL $(wc -l < $D/fail)"; head -c 300 $D/fail; command grep -a error $D/compile-arm.err | head -5; return 1; }
    ( cd $D && $PIN bash -e link.txt > link.log 2>&1 ) || { echo "$arm LINK-FAIL"; tail -5 $D/link.log; return 1; }
    cp $D/$exe $V/$arm.full && $OC $KS $V/$arm.full $V/$arm.strip.dom || return 1
    python3 $SCAN $V/$arm.full > $V/$arm.scan 2>&1; local sc=$?
    echo "$arm $(wc -l < $D/compile-arm.txt) compiles; strip $(sha256sum < $V/$arm.strip.dom | cut -c1-16) full $(sha256sum < $V/$arm.full | cut -c1-16); scan rc=$sc $(tail -1 $V/$arm.scan | command grep -a -o 'CDERIVE=[0-9]*\|CRENEW=[0-9]*\|CREVOKE=[0-9]*' | tr '\n' ' ')"
  done
  echo GCC-V2-DONE
}
main "$@"; exit
