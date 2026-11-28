#!/bin/bash
# build-renew.sh (nested lane, 2026-11-26): perlbench's CRENEW variant on Line B, built as build-ports.sh builds perl-full
# (B5TL1's own tree, its compile.txt and link.txt, llvm-c9, the B5TL1 SDK, the one-operand shim and inc/), plus three
# experiment scripts (renew/, never port scripts): heads-renew.py, bodies-renew-lb.py and he-renew-lb.py, each failing
# closed (a CRENEW that returns 0 falls back to the port's revoke). Arms:
#   perl-fullren0  perl-full's sources plus the three scripts, perl-full's defines only: the replay control; its image
#                  must be byte-identical to perl-full's (exes/perl-full.full), so the scripts change nothing unless asked
#   perl-fullren   the same tree, plus -DPERL_CAPSTONE_SUBLET_HEADS_RENEW -DPERL_CAPSTONE_SUBLET_BODIES_RENEW
#                  -DPERL_CAPSTONE_SUBLET_HE_RENEW and -include renew/renew-shim.h (PERL_CAP_RENEW = CRENEW)
#   perl-fullren-ff  perl-fullren plus -DPERL_CAPSTONE_RENEW_FORCE_FAIL: every renewal reads as failed (0, the old
#                  lifetime left live), so every site takes its fall-back; the fixtures' check of that path
# Op slabs are not renewed (oneop read that as a cost). Each patched tree must equal perl-full's except sv.c and
# capstone_sv_segs.h; each image is scanned with scan_illegal.py.
main() {
  set -u
  for v in $(env | sed -n 's/^\(CAPSTONE_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  export CAPSTONE_LLVM_BIN=<hdd>/capstone/llvm-c9/bin
  local X=<hdd>/capstone/nested/b5b2/ports T=<hdd>/capstone/perf/cclever/arms/B5TL1
  local OC=<hdd>/capstone/llvm-c4/bin/llvm-objcopy SCAN=<hdd>/capstone/nested/b5b2/scan_illegal.py
  local KS="--strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq"
  local PIN="taskset -c 16-27,48-59 nice -n 10" R=$X/renew
  { echo "compiler $($CAPSTONE_LLVM_BIN/clang --version | head -1)"; sha256sum $CAPSTONE_LLVM_BIN/clang $T/sdk/capstone-cc $X/oneop-shim.h \
      $X/inc/capstone/* $R/heads-renew.py $R/bodies-renew-lb.py $R/he-renew-lb.py $R/renew-shim.h $X/exes/perl-full.full; } > $R/inputs.txt
  for arm in ${ARMS:-perl-fullren0 perl-fullren perl-fullren-ff}; do
    local D=$X/$arm exe=perlbench_r
    [ -e $D ] && { echo "$D exists"; return 2; }
    cp -a $X/perl-full $D && find $D -name '*.o' -type f -delete && rm -f $D/$exe $D/fail $D/compile-arm.err $D/compile-arm.txt || return 1
    { python3 $R/heads-renew.py $D && python3 $R/bodies-renew-lb.py $D && python3 $R/he-renew-lb.py $D; } > $D/renew.log 2>&1 || { echo "$arm RENEW-FAIL"; tail -3 $D/renew.log; return 1; }
    local nd; nd=$(diff -rq $D $X/perl-full | command grep -a -v -E '\.o |\.o$|perlbench_r|\.log|fail|compile-arm' | command grep -a -v -E '/(sv\.c|capstone_sv_segs\.h) ' | wc -l)
    [ "$nd" = 0 ] || { echo "$arm: $nd files differ from perl-full beyond sv.c and capstone_sv_segs.h"; diff -rq $D $X/perl-full | head; return 1; }
    local defs="-DPERL_CAPSTONE_SUBLET -DPERL_CAPSTONE_SUBLET_BODIES -DPERL_CAPSTONE_SUBLET_HE -DPERL_CAPSTONE_SUBLET_OPS"
    case $arm in perl-fullren|perl-fullren-ff)
      defs="$defs -DPERL_CAPSTONE_SUBLET_HEADS_RENEW -DPERL_CAPSTONE_SUBLET_BODIES_RENEW -DPERL_CAPSTONE_SUBLET_HE_RENEW";; esac
    [ $arm = perl-fullren-ff ] && defs="$defs -DPERL_CAPSTONE_RENEW_FORCE_FAIL"
    defs="$defs -include $X/oneop-shim.h -I$X/inc"
    [ $arm = perl-fullren0 ] || defs="$defs -include $R/renew-shim.h"
    python3 - $D "$defs" <<'PY' || return 1
import sys
d, defs = sys.argv[1:3]
L = open(d + '/compile.txt').read().splitlines()
out = []
for l in L:
    if not l.strip(): continue
    assert l.count(' -c -o ') == 1, l[:120]
    out.append(l.replace(' -c -o ', ' ' + defs + ' -c -o ', 1))
open(d + '/compile-arm.txt', 'w').write('\n'.join(out) + '\n')
print('compile-arm.txt', len(out), 'lines, defs [%s]' % defs)
PY
    ( cd $D && $PIN xargs -a compile-arm.txt -d '\n' -P 11 -n 1 bash -c 'eval "$1" > /dev/null 2>> compile-arm.err || echo "FAIL $1"' _ > fail )
    [ -s $D/fail ] && { echo "$arm COMPILE-FAIL $(wc -l < $D/fail)"; head -c 300 $D/fail; command grep -a error $D/compile-arm.err | head -5; return 1; }
    ( cd $D && $PIN bash -e link.txt > link.log 2>&1 ) || { echo "$arm LINK-FAIL"; tail -5 $D/link.log; return 1; }
    cp $D/$exe $X/exes/$arm.full && $OC $KS $X/exes/$arm.full $X/exes/$arm.strip.dom || return 1
    python3 $SCAN $X/exes/$arm.full > $X/exes/$arm.scan 2>&1; local sc=$?
    echo "$arm $(wc -l < $D/compile-arm.txt) compiles; strip $(sha256sum < $X/exes/$arm.strip.dom | cut -c1-16) full $(sha256sum < $X/exes/$arm.full | cut -c1-16); scan rc=$sc $(tail -1 $X/exes/$arm.scan | cut -c1-120)"
  done
  if cmp -s $X/exes/perl-fullren0.full $X/exes/perl-full.full; then echo "REPLAY perl-fullren0 == perl-full"; else echo "REPLAY-DIFFERS perl-fullren0 != perl-full"; fi
  echo RENEW-DONE
}
main "$@"; exit
