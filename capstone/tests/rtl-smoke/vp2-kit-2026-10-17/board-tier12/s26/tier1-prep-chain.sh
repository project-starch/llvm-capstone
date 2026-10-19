#!/bin/bash
# After the tree A Linux steps end TAL DONE, in order (2026-10-19):
#  1. S-26's pair image: rev7's Tier 1 stage with ONLY vm/vm-contract.sh swapped for the pair's bounded, counter-reading
#     one; baked (the Tier 1 bake, under the buildroot bake lock); its Image and dtb kept as /tmp/capstone/s26/image-s26.
#  2. The stage restored exactly to rev7's record (stage.SHA256SUMS compared), which rev8's staging then requires.
#  3. The pair's three firmwares on that Image, from wrapper aab63d5 (wt-wrap-s26) and monitor ba9a0de (csbi-s26),
#     differing only in defines: cur (QNT0 print), var (+CAPSTONE_BOOT_QUANTUM=2000000), mir (+CAPSTONE_MIRROR_QUANTUM=1000).
#  4. rev8's stage (node_flat_records=4096) and build, as rev8-prep.sh did.
# Usage: tier1-prep-chain.sh <TAL out> <log>. Ends by itself. S26_T (the Tier 1 root), S26_BAKE, S26_FW, S26_REV8 replace
# the stage root and the three external steps for the stubbed dry run.
main() {
  set -u
  local W=${1:?TAL out} O=${2:?log} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local T=${S26_T:-/tmp/capstone/tier1} G=$S/.. X=/tmp/capstone/s26 I=/tmp/capstone/b0/fpga-image-tier1 end=$(( $(date +%s) + 86400 ))
  local BAKE=${S26_BAKE:-bash /tmp/capstone/b0/b0-bake-tier1.sh} FW=${S26_FW:-bash /tmp/capstone/r55-table/build-fpga-fw.sh}
  local D="-DCAPSTONE_SUPERVISED_CALL -DCAPSTONE_SUPERVISOR_CSR_EVENTS -DCAPSTONE_SUPERVISE_QUIET -DCAPSTONE_VP2_CSMINT -DCAPSTONE_CPMX_DIAG -DCAPSTONE_VP2_FETCH_RETRY -DCAPSTONE_QUANTUM_PRINT"
  until grep -a -q "^TAL \(DONE\|STOP\|HOLD\)" "$W" 2>/dev/null || [ "$(date +%s)" -ge $end ]; do sleep 60; done
  grep -a -q "^TAL DONE" "$W" || { echo "PREP HOLD: the tree A Linux steps did not end DONE ($(grep -a '^TAL' "$W" | tail -1))" >> "$O"; return 1; }
  # 1. the S-26 stage and bake
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "PREP STOP: the stage is not its record" >> "$O"; return 1; }
  cmp -s $T/stage/vm/vm-contract.sh $G/rev7/vm-contract.sh || { echo "PREP STOP: the staged vm-contract.sh is not rev7's" >> "$O"; return 1; }
  [ -e $X/image-s26 ] && { echo "PREP STOP: $X/image-s26 exists" >> "$O"; return 1; }
  cp $T/stage.SHA256SUMS $X/stage-rev7.SHA256SUMS; cp $S/vm-contract.sh $T/stage/vm/vm-contract.sh
  echo "PREP S26 BAKE $(date +%T): vm-contract.sh $(sha256sum $T/stage/vm/vm-contract.sh | cut -c1-16)" >> "$O"
  $BAKE > $X/bake-s26.out 2>&1; local rc=$?
  cp $G/rev7/vm-contract.sh $T/stage/vm/vm-contract.sh; cp $X/stage-rev7.SHA256SUMS $T/stage.SHA256SUMS   # restore first, whatever the bake did
  ( cd $T/stage && sha256sum -c --quiet $T/stage.SHA256SUMS ) || { echo "PREP STOP: the stage did not restore to rev7's record" >> "$O"; return 1; }
  echo "PREP stage restored to rev7's record $(date +%T)" >> "$O"
  [ $rc -eq 0 ] && grep -q BAKE_DONE $X/bake-s26.out && grep -q "cpio test-domains/vm/vm-contract.sh: present" $X/bake-s26.out \
    || { echo "PREP STOP: the S-26 bake failed (rc $rc): $(grep -E 'BAKE_DONE|FAILED|STOP' $X/bake-s26.out | tail -2 | tr '\n' ' ')" >> "$O"; return 1; }
  mkdir -p $X/image-s26; cp $I/Image $I/caplifive.dtb $X/image-s26/; chmod a-w $X/image-s26/*
  echo "PREP S26 IMAGE $(sha256sum $X/image-s26/Image | cut -c1-16) dtb $(sha256sum $X/image-s26/caplifive.dtb | cut -c1-16)" >> "$O"
  # 3. the three firmwares
  local a x
  for a in cur var mir; do
    case $a in cur) x="$D";; var) x="$D -DCAPSTONE_BOOT_QUANTUM=2000000";; mir) x="$D -DCAPSTONE_MIRROR_QUANTUM=1000";; esac
    [ -e $X/fw-s26-$a ] && { echo "PREP STOP: $X/fw-s26-$a exists" >> "$O"; return 1; }
    FW_WRAP=/tmp/capstone/wt-wrap-s26 FW_MON=/tmp/capstone/csbi-s26 FW_IMAGE_DIR=$X/image-s26 nice -n 10 $FW $X/fw-s26-$a "$x" > $X/fw-s26-$a.log 2>&1
    grep -q "^gate: PASS" $X/fw-s26-$a.log && [ -f $X/fw-s26-$a/fw_payload.bin ] || { echo "PREP STOP: firmware $a failed" >> "$O"; return 1; }
    echo "PREP S26 FW $a $(sha256sum $X/fw-s26-$a/fw_payload.bin | cut -c1-16) $(stat -c %s $X/fw-s26-$a/fw_payload.bin) B" >> "$O"
  done
  # 4. rev8
  echo "PREP REV8 $(date +%T)" >> "$O"
  ${S26_REV8:-bash $G/rev8/stage-rev8.sh 4096} >> "$O" 2>&1 || { echo "PREP STOP: rev8 staging failed" >> "$O"; return 1; }
  ${S26_REV8B:-bash $G/rev8/tier12-build-rev8.sh} >> "$O" 2>&1
  grep -q TIER12_BUILD_REV8_DONE $T/BUILD-REV8.log 2>/dev/null || { echo "PREP STOP: rev8's build failed" >> "$O"; return 1; }
  echo "PREP REV8 BUILT: $(grep '^fw_payload' $T/BUILD-REV8.log | tail -1)" >> "$O"
  echo "PREP DONE $(date +%T)" >> "$O"
}
main "$@"; exit $?
