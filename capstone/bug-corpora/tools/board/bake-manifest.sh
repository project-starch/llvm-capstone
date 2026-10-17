#!/bin/bash
# bake-manifest.sh <manifest> <image out dir>: bake a PRIVATE FPGA Linux image carrying exactly the manifest's files,
# then restore the shared buildroot (the Tier 3 board recipe's bake; generalized from the vp2 Tier 1 bake, which listed
# its files inline).
#
# The manifest has one `src|dst|mode` line per file, with dst relative to the overlay root (e.g.
# test-domains/vbatch/gate.sh). The rules are the Tier 1 bake's:
# - files are staged by explicit name, never a glob, into BOTH overlay/ and build-fpga/target/ (buildroot packs the
#   target);
# - a destination that already exists is refused (not ours to overwrite);
# - the image is rebuilt under the machine memory lock;
# - every staged file must be present in the new initramfs BY HASH;
# - Image + dtb are saved to <image out dir>;
# - exactly the manifest's files are removed and the shared image is rebuilt in an EXIT trap, then checked to carry
#   none of them.
# BAKE_REPLACE_KO=<path>: also replace the target's /capstone.ko (the board controls' process-ABI module) for this image,
# restored the same way.
main() {
  set -euo pipefail
  local MAN=${1:?manifest} OUT=${2:?image out dir}
  local FB=${BOARD_BUILDROOT:-$HOME/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot}
  local CC=${BOARD_CAPSTONE_CC:-$HOME/dev/llvm-capstone/capstone/capstone-c}
  local LOCK=${BOARD_MEMORY_LOCK:-$HOME/bin/logs/machine-memory.lock}
  [ -s "$MAN" ] || { echo "STOP: empty or missing manifest $MAN"; return 2; }
  [ -e "$OUT" ] && { echo "STOP: $OUT exists; a bake writes a fresh directory"; return 2; }
  mkdir -p "$OUT"; : > "$OUT/manifest.txt"
  local src dst mode root
  while IFS='|' read -r src dst mode; do
    [ -n "$src" ] || continue
    [ -f "$src" ] || { echo "STOP: missing $src"; return 2; }
    case "$dst" in /*|*..*) echo "STOP: destination must be relative and inside the overlay: $dst"; return 2;; esac
    for root in "$FB/overlay" "$FB/build-fpga/target"; do
      [ -e "$root/$dst" ] && { echo "REFUSING: $root/$dst already exists (not ours to overwrite)"; return 2; }
    done
  done < "$MAN"
  trap 'restore_files "$FB" "$OUT"' EXIT
  while IFS='|' read -r src dst mode; do
    [ -n "$src" ] || continue
    for root in "$FB/overlay" "$FB/build-fpga/target"; do
      mkdir -p "$(dirname "$root/$dst")"; cp "$src" "$root/$dst"; chmod "$mode" "$root/$dst"
    done
    echo "$(sha256sum "$src" | cut -d' ' -f1) $dst" >> "$OUT/manifest.txt"
  done < "$MAN"
  if [ -n "${BAKE_REPLACE_KO:-}" ]; then
    cp "$FB/build-fpga/target/capstone.ko" "$OUT/capstone.ko.stock"
    cp "$BAKE_REPLACE_KO" "$FB/build-fpga/target/capstone.ko"
  fi
  echo "$(date +%T) staged $(wc -l < "$OUT/manifest.txt") files; baking"
  flock -w 7200 "$LOCK" bash -c 'cd "$1" && nice -n 10 ionice -c3 make TARGET=fpga build LINUX_PAYLOAD=1 A=linux-rebuild CAPSTONE_CC_PATH="$2"' \
    _ "$FB" "$CC" > "$OUT/bake-run.log" 2>&1 || { echo "STOP: bake failed"; tail -5 "$OUT/bake-run.log"; return 1; }
  check_cpio "$FB" "$OUT" present
  if [ -n "${BAKE_REPLACE_KO:-}" ]; then  # the replaced /capstone.ko, by hash, as the vp2 Tier 1 bake checked it
    check_ko "$FB" "$OUT" "$(sha256sum "$BAKE_REPLACE_KO" | cut -d' ' -f1)" replaced
  fi
  cp "$FB/build-fpga/images/Image" "$FB/build-fpga/images/caplifive.dtb" "$OUT/"
  echo "$(date +%T) saved Image $(sha256sum "$OUT/Image" | cut -c1-16) dtb $(sha256sum "$OUT/caplifive.dtb" | cut -c1-16)"
  restore_files "$FB" "$OUT"; trap - EXIT
  flock -w 7200 "$LOCK" bash -c 'cd "$1" && nice -n 10 ionice -c3 make TARGET=fpga build LINUX_PAYLOAD=1 A=linux-rebuild CAPSTONE_CC_PATH="$2"' \
    _ "$FB" "$CC" > "$OUT/bake-restore.log" 2>&1 || { echo "STOP: RESTORE BAKE FAILED -- the shared image still carries the staged files"; return 1; }
  check_cpio "$FB" "$OUT" absent
  if [ -f "$OUT/capstone.ko.stock" ]; then
    check_ko "$FB" "$OUT" "$(sha256sum "$OUT/capstone.ko.stock" | cut -d' ' -f1)" stock
  fi
  echo "$(date +%T) shared image restored; BAKE_DONE"
}
restore_files() {  # <buildroot> <out>: remove exactly the manifest's files; put the stock /capstone.ko back
  local FB=$1 OUT=$2 h dst
  while read -r h dst; do rm -f "$FB/overlay/$dst" "$FB/build-fpga/target/$dst"; done < "$OUT/manifest.txt"
  [ -f "$OUT/capstone.ko.stock" ] && cp "$OUT/capstone.ko.stock" "$FB/build-fpga/target/capstone.ko"
  return 0
}
check_cpio() {  # <buildroot> <out> present|absent: hash each manifest file inside the freshly built initramfs
  local FB=$1 OUT=$2 want=$3 X=$2/cpio-x n=0 bad=0 h dst got
  rm -rf "${X:?}"; mkdir -p "$X"
  (cd "$X" && cpio -idm --quiet --no-absolute-filenames < "$FB/build-fpga/images/rootfs.cpio") 2>/dev/null || true
  while read -r h dst; do
    n=$((n+1))
    if [ -f "$X/$dst" ] && [ "$(sha256sum "$X/$dst" | cut -d' ' -f1)" = "$h" ]; then got=present; else got=absent; fi
    [ "$got" = "$want" ] || { echo "  cpio $dst: $got (want $want)"; bad=$((bad+1)); }
  done < "$OUT/manifest.txt"
  rm -rf "${X:?}"
  [ $n -gt 0 ] && [ $bad -eq 0 ] || { echo "STOP: CPIO CHECK FAILED ($want): $bad of $n"; return 1; }
  echo "  cpio: all $n manifest files $want"
}
check_ko() {  # <buildroot> <out> <want sha256> <label>: the initramfs's /capstone.ko must hash to <want>
  local FB=$1 OUT=$2 want=$3 X=$2/cpio-ko got
  rm -rf "${X:?}"; mkdir -p "$X"
  (cd "$X" && cpio -idm --quiet --no-absolute-filenames capstone.ko < "$FB/build-fpga/images/rootfs.cpio") 2>/dev/null || true
  got=$(sha256sum "$X/capstone.ko" 2>/dev/null | cut -d' ' -f1); rm -rf "${X:?}"
  [ "$got" = "$want" ] || { echo "STOP: cpio /capstone.ko is ${got:-absent}, not the $4 module $want"; return 1; }
  echo "  cpio /capstone.ko: the $4 module"
}
main "$@"; exit $?
