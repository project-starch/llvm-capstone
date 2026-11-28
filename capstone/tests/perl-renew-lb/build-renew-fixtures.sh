#!/bin/bash
# build-renew-fixtures.sh (nested lane, 2026-11-26): the protection fixtures of the Line B CRENEW variant, built exactly as
# build-fixtures.sh builds perl-full's: each committed make-{heads,bodies,he}-fixture.py generated from the tree's own
# perlmain.c, compiled with the tree's perlmain.o line from compile-arm.txt (B5TL1's line plus the arm's defines),
# linked in perlmain.o's place with link.txt, and stripped as the images are. Builds:
#   full-heads, stock-heads    perl-full's heads fixture (not built before on Line B) and its control on perl-stock,
#                              whose heads carry no lifetimes
#   fullren-{heads,bodies,he}  the variant
#   fullrenff-{heads,bodies,he} the forced-failure arm: every renewal reads as failed, so every site takes the port's
#                              revoke; it must fault where perl-full faults
# perl-full's bodies and HE fixtures and their NOREVOKE controls are fixture/bin's (build-fixtures.sh), reused unchanged.
main() {
  set -u
  for v in $(env | sed -n 's/^\(CAPSTONE_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  export CAPSTONE_LLVM_BIN=<hdd>/capstone/llvm-c9/bin
  local X=<hdd>/capstone/nested/b5b2/ports P=<hdd>/capstone/wt-nested/capstone/ports/spec-cpu2017/own-allocators/500.perlbench_r
  local OC=<hdd>/capstone/llvm-c4/bin/llvm-objcopy PIN="taskset -c 16-27,48-59 nice -n 10" F=$X/renew/fixture
  local KS="--strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq"
  [ -e $F ] && { echo "$F exists"; return 2; }
  mkdir -p $F/bin
  sha256sum $P/make-heads-fixture.py $P/make-bodies-fixture.py $P/make-he-fixture.py > $F/fixtures.sha256
  local spec
  for spec in full:perl-full:heads stock:perl-stock:heads fullren:perl-fullren:heads fullren:perl-fullren:bodies fullren:perl-fullren:he \
              fullrenff:perl-fullren-ff:heads fullrenff:perl-fullren-ff:bodies fullrenff:perl-fullren-ff:he; do
    local t=${spec%%:*} rest=${spec#*:}; local tree=${rest%%:*} fx=${rest#*:}; local D=$X/$tree
    [ -e $D/fx-$fx ] && { echo "$tree already has fx-$fx"; return 2; }
    python3 $P/make-$fx-fixture.py $D/perlmain.c $D/fx-$fx.c > /dev/null || { echo "$t $fx GEN-FAIL"; return 1; }
    command grep -a ' -o perlmain\.o ' $D/compile-arm.txt | sed "s| -o perlmain\.o | -o fx-$fx.o |; s| perlmain\.c *\$| fx-$fx.c|" > $D/fx-$fx-compile.cmd
    [ "$(wc -l < $D/fx-$fx-compile.cmd)" = 1 ] && command grep -q " fx-$fx\.c\$" $D/fx-$fx-compile.cmd || { echo "$t $fx: no perlmain compile line"; return 1; }
    sed "s| perlmain\.o | fx-$fx.o |; s| -o perlbench_r *\$| -o fx-$fx|" $D/link.txt > $D/fx-$fx-link.cmd
    command grep -q " fx-$fx\.o " $D/fx-$fx-link.cmd && command grep -q " -o fx-$fx\$" $D/fx-$fx-link.cmd || { echo "$t $fx: link edit failed"; return 1; }
    ( cd $D && $PIN sh fx-$fx-compile.cmd > fx-$fx-compile.log 2>&1 && $PIN sh fx-$fx-link.cmd > fx-$fx-link.log 2>&1 ) || { echo "$t $fx FX-BUILD-FAIL"; command grep -a error $D/fx-$fx-compile.log $D/fx-$fx-link.log | head -5; return 1; }
    $OC $KS $D/fx-$fx $F/bin/$t-$fx || return 1
    python3 <hdd>/capstone/nested/b5b2/scan_illegal.py $D/fx-$fx > $F/bin/$t-$fx.scan 2>&1; local sc=$?
    echo "fx $t-$fx strip $(sha256sum < $F/bin/$t-$fx | cut -c1-16) scan rc=$sc $(tail -1 $F/bin/$t-$fx.scan | command grep -a -o 'CDERIVE=[0-9]*\|CRENEW=[0-9]*\|CREVOKE=[0-9]*' | tr '\n' ' ')"
  done
  echo FIXTURES-DONE
}
main "$@"; exit
