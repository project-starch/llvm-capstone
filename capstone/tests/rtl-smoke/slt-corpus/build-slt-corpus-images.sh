#!/usr/bin/env bash
# Build the SQLLogicTest-corpus SQLite silicon images (one per region/heap class), validate each
# under QEMU on its own corpus file against the native baseline, and copy the validated pair
# (domain + host) to a DURABLE directory -- /tmp does not survive a reboot (2026-09-07 lost a
# campaign's images that way).
#
#   usage: build-slt-corpus-images.sh <images-dir> <native.txt>
#
# Classes (measured 2026-09-05 under QEMU; the region's top half holds the input, the heap is the
# SQLite memsys5 arena, the stack must keep the data allocation under the kernel's order-10 limit):
#   1m : heap 256 KiB, stack 2 MiB, region 1 MiB  -> select1, select2, negative control, aggfunc
#   s3 : heap 1 MiB,   stack 2 MiB, region 2 MiB  -> select3
#   s4 : heap 1 MiB,   stack 2 MiB, region 4 MiB  -> select4
#   s5 : heap 2 MiB,   stack 1 MiB, region 2 MiB  -> select5 (2 MiB stack does not fit: order 11)
# Serialised under the QEMU lock. Requires a FRESH toolchain (capstone-test-env.sh warns; make it
# fatal here so a stale binary cannot produce an image).
set -euo pipefail
IMG=${1:?images dir}; NATIVE=${2:?native baseline tsv}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
export CAPSTONE_REQUIRE_FRESH_TOOLCHAIN=1
source "$ROOT/capstone/tests/capstone-test-env.sh"
CORPUS=$(bash "$ROOT/capstone/benchmarks/sqlite/fetch-sqllogictest.sh" | tail -1)
mkdir -p "$IMG"; LOG="$IMG/build.log"; : > "$LOG"
exec 9>"${CAPSTONE_QEMU_LOCK:?source capstone-test-env.sh first: CAPSTONE_QEMU_LOCK is the one QEMU lock path}"; flock 9; export CAPSTONE_QEMU_LOCK_HELD=1
#      tag:heap:stack:region:validate-on:dom-name:host-name
for spec in "1m:262144:2097152:1048576:select2:sqslt1m.dom:sqlite_host_1m.user" \
            "s3:1048576:2097152:2097152:select3:sqslt3.dom:sqlite_host_2m.user" \
            "s4:1048576:2097152:4194304:select4:sqslt4.dom:sqlite_host_4m.user" \
            "s5:2097152:1048576:2097152:select5:sqslt5.dom:sqlite_host_2m5.user"; do
  IFS=: read -r tag heap stack region f dom host <<< "$spec"
  echo "== $(date +%T) $tag: heap $heap stack $stack region $region, validate on $f ==" >> "$LOG"
  OUT=/tmp/capstone/sqlite-slt-$tag
  set +e
  SQLITE_HEAP_SIZE=$heap SQLITE_SILICON_STACK=$stack SLT_REGION_SIZE=$region OUT_DIR=$OUT SHARE_DIR=/tmp/capstone/sqlite-slt-share-$tag SLT_TEST=$CORPUS/$f.test \
    timeout 5400 bash "$ROOT/capstone/benchmarks/sqlite/run-sqlite-slt.sh" > "$IMG/qemu-$tag.log" 2>&1; rc=$?
  set -e
  sil=$(grep -a -o 'SLT-SUMMARY.*' "$OUT/sqlite-slt.log" 2>/dev/null | tail -1 || true)
  nat=$(awk -F'\t' -v n="$f" '$1==n{print $2}' "$NATIVE")
  ok=no; [ -n "$sil" ] && [ -n "$nat" ] && [ "${sil%% completed*}" = "${nat%% completed*}" ] && ok=yes
  echo "rc=$rc match_native=$ok :: $sil" >> "$LOG"
  if [ "$ok" = yes ]; then
    cp -f "$OUT/sqlite_silicon.dom" "$IMG/$dom"; cp -f "$OUT/sqlite_host.user" "$IMG/$host"
    echo "staged $dom $(sha256sum "$IMG/$dom" | cut -c1-16) $host $(sha256sum "$IMG/$host" | cut -c1-16)" >> "$LOG"
  else echo "NOT staged: $tag ($f did not match native under QEMU)" >> "$LOG"; fi
done
echo "toolchain: $(sha256sum "$ROOT/llvm/cmake-build-debug/lib/libLLVMCapstoneCodeGen.so" | cut -c1-16) $(stat -c %y "$ROOT/llvm/cmake-build-debug/lib/libLLVMCapstoneCodeGen.so" | cut -c1-19)" >> "$LOG"
echo "$(date +%T) DONE" > "$IMG/build.marker"
