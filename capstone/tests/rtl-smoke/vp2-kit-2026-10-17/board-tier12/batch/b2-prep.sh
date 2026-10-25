#!/bin/bash
# Bitstream 2's (18-bit index) Tier 1 image and firmware (2026-10-23, before the bitstream seals): batch-prep.sh with ONE
# change, the staged module is the runtime lane's 18-bit board build 215a09be5a89360c (runtime/flat-index-18-gen 9e7bf6e0b584
# = the batch module 21fbfc39 plus -DCAPSTONE_VM_INDEX_BITS=18; rv64g, 0 RVC), output batch-t1-b2. batch-prep.sh's header
# follows unchanged. The batch bitstream's Tier 1 image and firmware, prepared before the bitstream (2026-10-21). Nothing here touches a board.
#  1. stage: a PRIVATE copy of rev7's Tier 1 stage (s26/stage-rev7.SHA256SUMS: the live stage's launcher, job runner and
#     vmrun.sh, and rev7's vm-contract.sh / vm-suite-inner.sh from stage-attic/rev8-1019-1626), those six files checked
#     against that record. The three images (contract.dom, mruby.dom, sqlite3.dom) come from <image dir>, NOT from
#     rev7: rev7's are the v9 kit's, built on runtime f6dfd19d9396, whose heap.c lacks 232d13b2's integers-first slot
#     offsets, so they fault at their first allocation on an R-60 bitstream (r60-runtime-gate.py REFUSE; found by the
#     compiler lane after the first prep, 2026-10-21). <runtime rev> must PASS that gate. vm/capstone_vm.ko is the
#     batch module 21fbfc397e7c7a42 (the runtime lane's
#     module-gen-push 64747600ecec, board build: rv64g, 0 compressed instructions, vermagic 6.4.14 SMP riscv);
#  2. image: the kits' manifest bake (bake-manifest.sh: the buildroot bake lock, by-hash initramfs check, the shared
#     buildroot restored) of base.manifest (the board controls) plus the stage under test-domains/, /capstone.ko
#     replaced as every vp2 kit does;
#  3. firmware: capstone-sbi vp2/batch d9da397 (f13dbc5 + the fetch-retry cap + RFENCE), wrapper aab63d5, the kits'
#     defines plus -DCAPSTONE_FETCH_RETRY_CAP; the dom_stack gate and the 26,000,000 B cap; the FRTC tag in the
#     generated monitor.
# Usage: batch-prep.sh <log> <image dir> <runtime rev> <sha256/16 of contract.dom,mruby.dom,sqlite3.dom>. Ends by itself. BP_BAKE / BP_FW stub the two external steps for the dry run; BP_X the root.
main() {
  set -u
  local O=${1:?log} IMG=${2:?image dir} REV=${3:?runtime rev} WANT=${4:?three image hashes} X=${BP_X:-/mnt/hdd/$USER/capstone/batch-t1-b2}
  local T=/tmp/capstone/tier1 A=/tmp/capstone/tier1/stage-attic/rev8-1019-1626 REC=/tmp/capstone/s26/stage-rev7.SHA256SUMS
  local K=/mnt/hdd/$USER/capstone/t3-smoke/bkit7-rev2xBDN2 P=/mnt/hdd/$USER/capstone/t3-smoke/bkit8-batch-parts
  local M=/mnt/hdd/$USER/capstone/board-modules/capstone_vm-215a09be5a89360c.ko MON=/tmp/capstone/csbi-batch WRAP=/tmp/capstone/wt-wrap-s26
  local BAKE=${BP_BAKE:-bash $K/bake-manifest.sh} FW=${BP_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local D; D="$(sed -n 's/^BOARD_FW_DEFINES="\(.*\)"$/\1/p' $K/board.env) -DCAPSTONE_FETCH_RETRY_CAP"
  case "$D" in *-DCAPSTONE_VP2_FETCH_RETRY*-DCAPSTONE_FETCH_RETRY_CAP) ;; *) echo "BP STOP: defines [$D]" >> "$O"; return 1;; esac
  [ "$(sha256sum $M | cut -c1-16)" = 215a09be5a89360c ] || { echo "BP STOP: the module is not 215a09be" >> "$O"; return 1; }
  [ "$(git -C $MON rev-parse --short=7 HEAD)" = d9da397 ] && [ -z "$(git -C $MON status --porcelain)" ] \
    && [ "$(git -C $WRAP rev-parse --short=7 HEAD)" = aab63d5 ] || { echo "BP STOP: monitor or wrapper not at d9da397 / aab63d5" >> "$O"; return 1; }
  local G=/tmp/capstone/wt-gate/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/r60-runtime-gate.py
  python3 $G --rev "$REV" >> "$O" 2>&1 || { echo "BP STOP: runtime $REV does not pass r60-runtime-gate" >> "$O"; return 1; }
  local got; got=$(for f in contract.dom mruby.dom sqlite3.dom; do sha256sum $IMG/$f 2>/dev/null | cut -c1-16; done | paste -sd, -)
  [ "$got" = "$WANT" ] || { echo "BP STOP: the images are $got, not $WANT" >> "$O"; return 1; }
  [ -e $X ] && { echo "BP STOP: $X exists" >> "$O"; return 1; }
  mkdir -p $X/stage/vm
  local f
  for f in vmrun.sh vm/capstone-job vm/capstone-vexec vm/capstone_vm.ko; do cp -p $T/stage/$f $X/stage/$f; done
  cp -p $A/vm/vm-contract.sh $A/vm/vm-suite-inner.sh $X/stage/vm/
  # the six non-image files must be rev7's record, line for line
  ( cd $X/stage && grep -v -E ' vm/(contract|mruby|sqlite3)\.dom$' $REC | sha256sum -c --quiet - ) \
    && [ "$(cd $X/stage && find . -type f | wc -l)" -eq 6 ] || { echo "BP STOP: the private stage's six files are not rev7's record" >> "$O"; return 1; }
  for f in contract.dom mruby.dom sqlite3.dom; do cp $IMG/$f $X/stage/vm/$f; done
  chmod u+w $X/stage/vm/capstone_vm.ko; cp $M $X/stage/vm/capstone_vm.ko
  [ "$(sha256sum $X/stage/vm/capstone_vm.ko | cut -c1-16)" = 215a09be5a89360c ] || { echo "BP STOP: the staged module is not 215a09be" >> "$O"; return 1; }
  ( cd $X/stage && sha256sum $(find . -type f | sed 's|^\./||' | sort) ) > $X/stage.SHA256SUMS
  echo "BP STAGE rev7's six files, images $WANT (runtime $REV, gate PASS), capstone_vm.ko 215a09be5a89360c ($(wc -l < $X/stage.SHA256SUMS) files)" >> "$O"
  cp $K/base.manifest $X/manifest.txt
  ( cd $X/stage && find . -type f | sed 's|^\./||' | sort ) | while read -r f; do
    case $f in *.sh|vm/capstone-job|vm/capstone-vexec) echo "$X/stage/$f|test-domains/$f|0755";; *) echo "$X/stage/$f|test-domains/$f|0644";; esac
  done >> $X/manifest.txt
  echo "BP BAKE $(date +%T): $(wc -l < $X/manifest.txt) files" >> "$O"
  BAKE_REPLACE_KO=/tmp/capstone/r55-table/fpga-pkg-127/module/capstone.ko $BAKE $X/manifest.txt $X/image > $X/bake.out 2>&1; local rc=$?
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake.out || { echo "BP STOP: the bake failed (rc $rc): $(tail -2 $X/bake.out | tr '\n' ' ')" >> "$O"; return 1; }
  echo "BP IMAGE $(sha256sum $X/image/Image | cut -c1-16) dtb $(sha256sum $X/image/caplifive.dtb | cut -c1-16)" >> "$O"
  FW_WRAP=$WRAP FW_MON=$MON FW_IMAGE_DIR=$X/image nice -n 10 $FW $X/fw "$D" > $X/fw.log 2>&1
  grep -q "^gate: PASS" $X/fw.log && [ -f $X/fw/fw_payload.bin ] || { echo "BP STOP: the firmware failed" >> "$O"; return 1; }
  local sz; sz=$(stat -c %s $X/fw/fw_payload.bin)
  [ "$sz" -le 26000000 ] || { echo "BP STOP: firmware $sz B over the cap" >> "$O"; return 1; }
  grep -q "0x46525443\|1179800643" $X/fw/opensbi/lib/sbi/sbi_capstone_dom.c.S || { echo "BP STOP: no FRTC tag in the generated monitor" >> "$O"; return 1; }
  echo "BP FW $(sha256sum $X/fw/fw_payload.bin | cut -c1-16) $sz B on image $(sha256sum $X/image/Image | cut -c1-16); FRTC present" >> "$O"
  echo "BP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
