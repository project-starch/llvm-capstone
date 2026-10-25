#!/bin/bash
# COPY of /tmp/capstone/b0/b0-bake-tier1.sh for the fetch-face pair (2026-10-20): the same bake, plus the pair's two images.
# vp2 Tier 1/2 (copy of b0-bake-bootp.sh, BL's image, plus /test-domains/vm and vmrun.sh): bake a PRIVATE FPGA Linux image, then restore the shared buildroot.
# Stages by explicit name with a manifest (never a glob), into BOTH overlay/ and build-fpga/target/ (buildroot packs
# the target), rebuilds the Image under the machine memory lock, verifies by HASH that the initramfs carries every
# staged file, saves Image + dtb, then removes exactly the manifest's files and rebuilds so other lanes get the stock
# image back -- verified the same way (none of the five present).
main() {
  set -euo pipefail
  # The buildroot bake lock (2026-10-19, board #2 phase 1): held for this WHOLE bake (stage, rebuild, save, restore),
  # so two bakes never share the one buildroot tree, whichever board each is for. The kernel drops a flock when its
  # fd closes, so any exit (an error, a kill, a crash) releases it; the memory lock below is only ever taken inside it.
  exec 7>>/tmp/capstone/.buildroot-bake.lock
  flock -w 14400 7 || { echo "STOP: the buildroot bake lock was not free within 4 h"; return 1; }
  local FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
  local CC=$HOME/dev/llvm-capstone/capstone/capstone-c
  local OUT=/tmp/capstone/b0/fpga-image-tier1 LOCK=$HOME/bin/logs/machine-memory.lock
  local W=/tmp/capstone/wt-b0/capstone/runtime/silicon
  # src|dst-relative-to-overlay|mode. Tier 1 keeps only BL's two controls (b0-stats, b0-hello) and their support;
  # BL's b5/r54/cw/b6 rungs are dropped to keep the firmware inside the 20 MB working budget.
  local M=/tmp/capstone/b5m/board
  local FILES=(
    "/tmp/capstone/b1/bake-hello.dom|test-domains/b0-hello.dom|0644"
    "/tmp/capstone/r55-table/fpga-pkg-127/module/capstone.ko|test-domains/capstone-proc.ko|0644"
    "$W/b0run.sh|test-domains/b0run.sh|0755"
    "/tmp/capstone/b0/fpga-exec/capstone-exec|usr/bin/capstone-exec|0755"
    "/tmp/capstone/b0/fpga-exec/capstone-job|usr/bin/capstone-job|0755"
    "/tmp/capstone/tier1/stage/vmrun.sh|test-domains/vmrun.sh|0755"
    "/tmp/capstone/tier1/stage/vm/capstone_vm.ko|test-domains/vm/capstone_vm.ko|0644"
    "/tmp/capstone/tier1/stage/vm/capstone-vexec|test-domains/vm/capstone-vexec|0755"
    "/tmp/capstone/tier1/stage/vm/capstone-job|test-domains/vm/capstone-job|0755"
    "/tmp/capstone/tier1/stage/vm/contract.dom|test-domains/vm/contract.dom|0644"
    "/tmp/capstone/tier1/stage/vm/sqlite3.dom|test-domains/vm/sqlite3.dom|0644"
    "/tmp/capstone/tier1/stage/vm/mruby.dom|test-domains/vm/mruby.dom|0644"
    "/tmp/capstone/tier1/stage/vm/vm-contract.sh|test-domains/vm/vm-contract.sh|0755"
    "/tmp/capstone/tier1/stage/vm/vm-suite-inner.sh|test-domains/vm/vm-suite-inner.sh|0755")
  # rev8 (2026-10-19) stages the compiler lane's dense-churn image beside contract.dom. Baked only when the stage holds it,
  # so every earlier stage bakes exactly as before.
  # The alias-load test (2026-10-20; board-tier12/alias): the compiler lane's one image (lane alias-load ae6a930bc0ab).
  # Only this copy of the Tier 1 bake stages it.
  FILES+=("/mnt/hdd/$USER/capstone/aliasload/alias-load.dom|test-domains/vm/alias-load.dom|0644")
  [ -f /tmp/capstone/tier1/stage/vm/contract-dense.dom ] && FILES+=("/tmp/capstone/tier1/stage/vm/contract-dense.dom|test-domains/vm/contract-dense.dom|0644")
  # k800r.dom dropped for attempt 5: a classic control cannot share a module load with the process ABI.
  mkdir -p "$OUT"; : > "$OUT/manifest.txt"
  # REPLACE (2026-10-04, attempt 3): the board drivers always run `[ -e /dev/capstone ] || insmod /capstone.ko`
  # before any rung (run_ladder_perf_fpga.py:193-202), and the board kernel cannot unload a module. So in THIS
  # private image /capstone.ko IS the process-ABI module. The shared target's stock file is backed up first and
  # put back by the EXIT trap; its hash is checked after the restore bake.
  local NEWKO=/tmp/capstone/r55-table/fpga-pkg-127/module/capstone.ko
  cp "$FB/build-fpga/target/capstone.ko" "$OUT/capstone.ko.stock"
  sha256sum "$OUT/capstone.ko.stock" | cut -d" " -f1 > "$OUT/capstone.ko.stock.sha"
  for e in "${FILES[@]}"; do
    IFS='|' read -r src dst mode <<< "$e"
    [ -f "$src" ] || { echo "missing $src"; return 2; }
    for root in "$FB/overlay" "$FB/build-fpga/target"; do
      [ -e "$root/$dst" ] && { echo "REFUSING: $root/$dst already exists (not ours to overwrite)"; return 2; }
    done
  done
  for e in "${FILES[@]}"; do
    IFS='|' read -r src dst mode <<< "$e"
    for root in "$FB/overlay" "$FB/build-fpga/target"; do
      mkdir -p "$(dirname "$root/$dst")"; cp "$src" "$root/$dst"; chmod "$mode" "$root/$dst"
    done
    echo "$(sha256sum "$src" | cut -d' ' -f1) $dst" >> "$OUT/manifest.txt"
  done
  cp "$NEWKO" "$FB/build-fpga/target/capstone.ko"
  echo "$(sha256sum "$NEWKO" | cut -d' ' -f1) capstone.ko" >> "$OUT/replaced.txt.new"; mv "$OUT/replaced.txt.new" "$OUT/replaced.txt"
  # RESTORE IN AN EXIT TRAP. The first version restored as a later step, and a non-root cpio exit (rc 2) under
  # set -e aborted between the bake and the restore, leaving the shared image carrying B0 files (2026-10-04).
  trap 'restore_files; echo "EXIT: staged files removed from overlay/target; rebake the shared image if it was baked"' EXIT
  echo "$(date +%T) staged $(wc -l < "$OUT/manifest.txt") files; baking"
  bake() {
    flock -w 7200 "$LOCK" bash -c 'cd "$1" && nice -n 10 ionice -c3 make TARGET=fpga build LINUX_PAYLOAD=1 A=linux-rebuild CAPSTONE_CC_PATH="$2"' _ "$FB" "$CC" > "$OUT/bake-$1.log" 2>&1
  }
  bake run || { echo "bake failed"; tail -5 "$OUT/bake-run.log"; restore_files; return 1; }
  check_cpio present
  cp "$FB/build-fpga/images/Image" "$FB/build-fpga/images/caplifive.dtb" "$OUT/"
  echo "$(date +%T) saved Image $(sha256sum "$OUT/Image" | cut -c1-16) dtb $(sha256sum "$OUT/caplifive.dtb" | cut -c1-16)"
  restore_files
  bake restore || { echo "RESTORE BAKE FAILED -- the shared image still carries B0 files"; return 1; }
  check_cpio absent
  echo "$(date +%T) shared image restored; BAKE_DONE"
}
restore_files() {
  local FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot
  while read -r h dst; do
    rm -f "$FB/overlay/$dst" "$FB/build-fpga/target/$dst"
  done < /tmp/capstone/b0/fpga-image-tier1/manifest.txt
  rmdir "$FB/overlay/usr/bin" "$FB/overlay/usr" "$FB/overlay/test-domains/mruby" "$FB/build-fpga/target/test-domains/mruby" 2>/dev/null || true
  rmdir "$FB/overlay/test-domains/vm" "$FB/build-fpga/target/test-domains/vm" 2>/dev/null || true
  if [ -f /tmp/capstone/b0/fpga-image-tier1/capstone.ko.stock ]; then
    cp /tmp/capstone/b0/fpga-image-tier1/capstone.ko.stock "$FB/build-fpga/target/capstone.ko"
  fi
}
check_cpio() {  # present | absent: hash each manifest file inside the freshly built initramfs
  local FB=$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot X=/tmp/capstone/b0/fpga-image-tier1/cpio-x
  rm -rf "$X"; mkdir -p "$X"
  # cpio exits non-zero as a non-root user (it cannot create the device nodes); only the five files matter
  (cd "$X" && cpio -idm --quiet --no-absolute-filenames < "$FB/build-fpga/images/rootfs.cpio") 2>/dev/null || true
  local n=0 bad=0
  while read -r h dst; do
    n=$((n+1))
    if [ -f "$X/$dst" ] && [ "$(sha256sum "$X/$dst" | cut -d' ' -f1)" = "$h" ]; then got=present; else got=absent; fi
    echo "  cpio $dst: $got"
    [ "$got" = "$1" ] || bad=$((bad+1))
  done < /tmp/capstone/b0/fpga-image-tier1/manifest.txt
  local want; if [ "$1" = present ]; then want=$(cut -d" " -f1 /tmp/capstone/b0/fpga-image-tier1/replaced.txt); else want=$(cat /tmp/capstone/b0/fpga-image-tier1/capstone.ko.stock.sha); fi
  if [ "$(sha256sum "$X/capstone.ko" | cut -d" " -f1)" = "$want" ]; then echo "  cpio /capstone.ko: as expected ($1)"; else echo "  cpio /capstone.ko: WRONG ($1)"; bad=$((bad+1)); fi
  [ $n -gt 0 ] && [ $bad -eq 0 ] || { echo "CPIO CHECK FAILED ($1): $bad of $n"; return 1; }
}
main "$@"; exit $?
