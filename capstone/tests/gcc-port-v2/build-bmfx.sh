#!/bin/bash
# build-bmfx.sh (nested lane, 2026-11-26): oneop's bitmap fixture (fixture/bm/bitmap-fixture.c, modes 0-3) on Line B,
# for gcc-port (v1, CFP-g3's image), gcc-port2 (v2) and gcc-port2ff: compiled with the arm's own bitmap.o compile line
# (source and object swapped), linked by the B5TL1 SDK's capstone-cc -O2 with that arm's bitmap.o and obstack.o only,
# stripped as the images are. In mode 2 bitmap_clear frees a chain of three elements through bitmap_elt_clear_from.
main() {
  set -u
  for v in $(env | sed -n 's/^\(CAPSTONE_[A-Z0-9_]*\)=.*/\1/p'); do unset "$v"; done
  export CAPSTONE_LLVM_BIN=<hdd>/capstone/llvm-c9/bin
  local V=<hdd>/capstone/nested/b5b2/ports/gcc-v2 X=<hdd>/capstone/nested/b5b2/ports
  local SRC=<hdd>/capstone/nested/oneop/fixture/bm/bitmap-fixture.c CC=<hdd>/capstone/perf/cclever/arms/B5TL1/sdk/capstone-cc
  local OC=<hdd>/capstone/llvm-c4/bin/llvm-objcopy SCAN=<hdd>/capstone/nested/b5b2/scan_illegal.py
  local KS="--strip-all --keep-section=.capstone_virtual --keep-section=.capstone_application --keep-section=.capstone_domreq"
  local F=$V/bmfx; [ -e $F ] && { echo "$F exists"; return 2; }; mkdir -p $F
  sha256sum $SRC > $F/source.sha256
  for spec in v1:$X/gcc-port v2:$V/gcc-port2 v2ff:$V/gcc-port2ff; do
    local n=${spec%%:*} D=${spec#*:}
    cp $SRC $D/bmfx.c
    command grep -a ' -o bitmap\.o ' $D/compile-arm.txt | sed 's| -o bitmap\.o | -o bmfx.o |; s| bitmap\.c *$| bmfx.c|' > $D/bmfx-compile.cmd
    [ "$(wc -l < $D/bmfx-compile.cmd)" = 1 ] && command grep -q ' bmfx\.c$' $D/bmfx-compile.cmd || { echo "$n: no bitmap.o line"; return 1; }
    ( cd $D && sh bmfx-compile.cmd > bmfx-compile.log 2>&1 && $CC -O2 bmfx.o bitmap.o obstack.o -o bmfx > bmfx-link.log 2>&1 ) || { echo "$n BMFX-BUILD-FAIL"; tail -5 $D/bmfx-compile.log $D/bmfx-link.log; return 1; }
    $OC $KS $D/bmfx $F/bmfx-$n || return 1
    python3 $SCAN $D/bmfx > $F/bmfx-$n.scan 2>&1; local sc=$?
    echo "bmfx-$n strip $(sha256sum < $F/bmfx-$n | cut -c1-16) bitmap.o $(sha256sum < $D/bitmap.o | cut -c1-16) scan rc=$sc $(tail -1 $F/bmfx-$n.scan | command grep -a -o 'CDERIVE=[0-9]*\|CRENEW=[0-9]*\|CREVOKE=[0-9]*' | tr '\n' ' ')"
  done
  echo BMFX-DONE
}
main "$@"; exit
