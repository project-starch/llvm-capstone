#!/bin/bash
# BITSTREAM 3 version (2026-10-28; pre-registered in README.md before its first boot): tier3-mp-perl-b2.sh with only the board side
# changed: the bitstream-3 kit bkit11-b3-b<board> (make-b3-kit.sh copies of the bitstream-2 kits, one board.env line,
# BOARD_BITSTREAM=caplifive_vp2c_e7af48457, module 215a09be unchanged), output <mnt-hdd>/capstone/t3-run-b3-e7af48457-
# b<board>/, and "bitstream 3" in the booking text. The references (the bitstream-2 twin's), images, runners, trees, SDK
# and grade are bitstream 2's, unchanged. The bitstream-2 version's header follows.
# BITSTREAM 2 version (2026-10-25; pre-registered in README.md before its boot): tier3-mp-perl-batch.sh on the board's
# bitstream-2 kit bkit9-b2-b<board> (caplifive_vp2c_dcc6daf6e, module 215a09be), scored against the compiler lane's
# bitstream-2 references batch-refs-b2/rest/<tag>-trim (qkit7-b2twin3-q8). The batch reference's stamp still pins the
# runner, arm script, tree, image and closure (the bitstream-2 run used the same inputs); batch-refs-b2/REST-A must name
# the twin (BEGIN ... qemu 708bbff5 module 33254735), carry this tag's STAMP line with the same runner, closure and image,
# a clean read and "differ 0", and the reference's collected images must equal the batch reference's.
# Output <mnt-hdd>/capstone/t3-run-b2-dcc6daf6e-b<board>/mp-<tag>. The batch version's header follows.
# perl's release-differential arms on the BATCH bitstream c721c1a2a (2026-10-23; pre-registered in README.md before the
# boots): tier3-mp-arm.sh's perl path with what the batch changes:
#   - the kit is the board's batch kit bkit8-batch-b<board> (c721c1a2a, module 21fbfc39, monitor d9da397);
#   - the reference is the compiler lane's TRIMMED batch reference batch-refs-r60/batch/<tag>-trim (qkit7-batch2: its stamp
#     names qemu ecaf50bbd5ec44aa and module a05c198bfcc5f55c; dev 120360c608dc, pre-registered at lane f83b8ffde608; full
#     = trimmed 11 of 11 per arm), with --lib-closure 2a472e21 (the board's staging);
#   - the images are the r60 rebuilds corpus-mp-r60/cw/<tag> (perl-stock 76f1141d, perl-sublet bec05c86), runtime 4a5e480c;
#   - the tree is wt-ref2 at the stamp's "tree" commit (15ced322c824), the runner and arm script by the stamp's hashes.
# Output to NEW directories <mnt-hdd>/capstone/t3-run-batch-c721c1a2a-b<board>/mp-<tag>. tier3-mp-arm.sh's header follows.
main() {
  set -u
  local BRD=${1:?board} O=${2:?out log}; shift 2
  local S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local H=/mnt/hdd/$USER/capstone C=/mnt/hdd/$USER/capstone/corpus-mp-r60 W LB=/tmp/capstone/vp2-prep/llvm-bin-v6 K bs RR BUSY
  case $BRD in
    1) K=$H/t3-smoke/bkit11-b3-b1; bs=caplifive_vp2c_e7af48457; RR=${MP_ROOT:-$H/t3-run-b3-e7af48457-b1/mp}; BUSY="python3 /tmp/capstone/wt-gate/capstone/tests/rtl-smoke/board-busy.py 1";;
    2) K=$H/t3-smoke/bkit11-b3-b2; bs=caplifive_vp2c_e7af48457; RR=${MP_ROOT:-$H/t3-run-b3-e7af48457-b2/mp}; BUSY="python3 /tmp/capstone/wt-b2/capstone/tests/rtl-smoke/board-busy.py 2";;
    *) echo "MP STOP: board $BRD" >> "$O"; return 2;;
  esac
  ( cd $K && sha256sum -c --quiet SHA256SUMS ) && grep -qx "BOARD_BITSTREAM=$bs" $K/board.env && [ "$(sha256sum $K/adapter/module/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] \
    && { [ $BRD = 2 ] && grep -qx "FPGA_BOARD=2" $K/board.env || { [ $BRD = 1 ] && ! grep -q "^FPGA_BOARD=2" $K/board.env; }; } || { echo "MP STOP: kit $K" >> "$O"; return 1; }
  local AP tag ref st run args wref rsha asha d rc bid B key ok
  for tag in "$@"; do
    case $tag in
      perl-stock|perl-sublet)
        W=$H/wt-ref2; key=tree; ref=${MP_REFS:-$H/batch-refs-b2/rest}/$tag-trim; bref=$H/batch-refs-r60/batch/$tag-trim; ok="controls {'eval': True, 'shim': True}"
        run=$W/capstone/bug-corpora/perl/release-differential/runners/virtual/run-virtual.py
        [ $tag = perl-stock ] && args="--arm virtual-malloc" || args="--arm virtual-nested-pools"
        args="$args --perl $C/cw/$tag --lib-closure $W/capstone/bug-corpora/perl/release-differential/runners/virtual/board/lib-closure.txt";;
      *) echo "MP STOP: tag $tag has no recipe" >> "$O"; return 2;;
    esac
    st=$bref.stamp; AP=$W/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
    [ -s $ref/out/matrix.tsv ] && [ -d $ref/session/collect/images ] || { echo "MP STOP: $tag has no reference bundle or collected images" >> "$O"; return 1; }
    grep -q "^STAMP $tag-trim batch .* qemu ecaf50bbd5ec44aa module a05c198bfcc5f55c .*image $(sha256sum $C/cw/$tag/src/perl-5.36.3/perl | cut -c1-16) .*staging TRIMMED lib-closure $(sha256sum $W/capstone/bug-corpora/perl/release-differential/runners/virtual/board/lib-closure.txt | cut -c1-16)" $st \
      || { echo "MP STOP: $tag's reference stamp is not the batch kit's trimmed run of this image and closure" >> "$O"; return 1; }
    # A reference whose own controls failed reads as all-fault, and a board run of the same broken image matches it.
    grep -qF "$ok" <(grep "^APPVM read rc=0 " $bref.log) || { echo "MP STOP: $tag's batch reference did not read clean (needs APPVM read rc=0 with \"$ok\" in $bref.log)" >> "$O"; return 1; }
    local RA=$H/batch-refs-b2/REST-A ish csha
    ish=$(sha256sum $C/cw/$tag/src/perl-5.36.3/perl | cut -c1-16); csha=$(sha256sum $W/capstone/bug-corpora/perl/release-differential/runners/virtual/board/lib-closure.txt | cut -c1-16)
    grep -q "^BEGIN A .* qemu 708bbff5663f0e86 module 3325473537072a3a " $RA && grep -q "^STAMP $tag-trim runner $(sha256sum $run | cut -c1-16) closure $csha image $ish " $RA \
      && grep -q "^$tag-trim rc=0 .*APPVM read rc=0 .*$ok" $RA && grep -q "$tag trimmed batch -> b2: cells 11/11 differ 0$" $RA \
      && diff -q <(cd $bref/session/collect/images && sha256sum *) <(cd $ref/session/collect/images && sha256sum *) > /dev/null \
      || { echo "MP STOP: the bitstream-2 reference $tag-trim is not the twin's clean run of the batch inputs" >> "$O"; return 1; }
    wref=$(grep -o "$key [0-9a-f]\{12\} dirty 0" $st | head -1 | cut -d' ' -f2); rsha=$(grep -o "runner [0-9a-f]\{16\}" $st | head -1 | cut -d' ' -f2); asha=$(grep -o "appvm-arm [0-9a-f]\{16\}" $st | head -1 | cut -d' ' -f2)
    [ -n "$wref" ] && [ "$(git -C $W rev-parse --short=12 HEAD)" = "$wref" ] && [ -z "$(git -C $W status --porcelain)" ] || { echo "MP STOP: $W is not the clean wt-ref commit $tag's stamp names (${wref:-none})" >> "$O"; return 1; }
    [ "$(sha256sum $run | cut -c1-16)" = "$rsha" ] && [ "$(sha256sum $AP | cut -c1-16)" = "$asha" ] || { echo "MP STOP: the runner or arm script is not $tag's reference's ($rsha / $asha)" >> "$O"; return 1; }
    d=$RR-$tag; [ -e "$d" ] && { echo "MP STOP: $d exists" >> "$O"; return 1; }
    mkdir -p "$(dirname $d)"
    ${MP_BUSY:-$BUSY} > /dev/null || { echo "MP STOP: board $BRD is busy" >> "$O"; return 1; }
    bid=$(${MP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} book $BRD 240 "vp2 bitstream 3 $tag (board $BRD)"); rc=$?
    [ $rc = 3 ] && { echo "MP STOP: the hub refused the booking (409): $bid" >> "$O"; return 1; }
    [ $rc = 0 ] || { echo "MP NOTE: booking failed rc=$rc ($bid); continuing unbooked" >> "$O"; bid=; }
    ( cd $ref/session/collect/images && sha256sum $(ls | sort) ) > $d.images.SHA256SUMS
    echo "MP $tag BEGIN $(date +%T) board $BRD runner $rsha appvm-arm $asha wt-ref $wref booking ${bid:-none}" >> "$O"
    # shellcheck disable=SC2086
    EXPECT_IMAGES=$d.images.SHA256SUMS taskset -c 16-31 timeout 18000 ${MP_ARM:-bash $AP} --board-kit $K $d $run $args --llvm-bin $LB > $d.run.log 2>&1; rc=$?
    [ -n "$bid" ] && ${MP_BOOK:-python3 /tmp/capstone/lab-hub/lab-book.py} unbook $bid >> "$O" 2>&1
    echo "MP $tag END $(date +%T) rc=$rc :: $(tail -1 $d.run.log | cut -c1-160)" >> "$O"
    case $rc in 78|2) echo "MP STOP: the arm script refused $tag (rc=$rc)" >> "$O"; return 1;; esac
    # The board run's own controls must pass too, by the same test as the reference's: otherwise no row is a reading,
    # whatever the compare prints (a bounded smoke reads FAILED and its cells can still match).
    if grep -qF "$ok" <(grep "^APPVM read rc=0 " $d.run.log); then
      python3 $S/matrix-compare.py $ref/out $d/out > $d.cmp.txt 2>&1; echo "MP $tag COMPARE vs reference rc=$? :: $(tail -1 $d.cmp.txt)" >> "$O"
    else
      echo "MP $tag VOID: the board run's own controls did not pass (needs APPVM read rc=0 with \"$ok\"): no cell is a reading :: $(grep '^APPVM read' $d.run.log | tail -1 | cut -c1-140)" >> "$O"
    fi
    B=$(find $d -name board.txt -path '*guest*' 2>/dev/null | head -1)
    if [ -n "$B" ] && [ -s "$B" ]; then
      echo "MP $tag readings: pc=0 $(grep -a -c 'pc=0x0\b\|pc=0 ' $B) DRT1 $(grep -a -c '^DRT1:' $B) FRTY $(grep -a -c 'FRTY:' $B) Oops $(grep -a -c Oops $B) INJS $(grep -a -c INJS $B) INJL $(grep -a -c INJL $B) banner node-floor-2 x$(grep -a -c node-floor-2 $B)" >> "$O"
    else echo "MP $tag readings: NO DATA, no guest board.txt" >> "$O"; fi
    [ -n "$B" ] && grep -a -q '^### RUNG [0-9]* vbatch END rc=0 ###' "$B" || { echo "MP STOP after $tag: the boot did not complete (no vbatch END rc=0)" >> "$O"; return 1; }
  done
  echo "MP board$BRD DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
