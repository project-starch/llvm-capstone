#!/usr/bin/env bash
# The whole microbenchmark pipeline for both arms, from SDKs to manifests, in one command.
#
#   pipeline.sh ROOT [SOURCE_TREE] [STAGE...]
#
# ROOT          output root: SDKs, builds (ROOT/build/{T1,T2,T1-count,T2-count,native,T2-default}),
#               QEMU runs (ROOT/run-*), manifests and tables (ROOT/*.tsv)
# SOURCE_TREE   the tree the SDKs are built from (default: this checkout); e.g. a worktree with a
#               runtime fix. Programs, harness and scripts always come from THIS checkout.
# STAGE         any of: sdks build size rebuild gate functional report (default: all, in order).
#               `size` boots the sizing/counting pass and rewrites sizes.tsv and kvariants.txt in
#               this checkout (the record of the chosen iteration counts); `rebuild` is `build` again.
# Environment: CAPSTONE_LLVM_BIN, CAPSTONE_LLVM_BUILD_DIR (the compiler); MUSL_CACHE (a directory
# holding musl-1.2.5.tar.gz); KIT (the QEMU kit: the bitstream-2 twin with bench counters);
# BENCH_NATIVE_CC (buildroot's riscv64-buildroot-linux-gnu-gcc);
# ARGS (W I for the functional pass, default "1 10", the pilot's per-process count);
# CAPSTONE_QEMU_LOCK (the shared QEMU lock). Heavy work runs on CPUs 16-23,48-55 with -j12, QEMU on
# 24-31,56-63, one QEMU at a time under the lock.
# The body is a function, called on the last line, so editing this file while a run is in
# progress cannot change what that run executes (bash reads a script as it goes).
main() {
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(mkdir -p "${1:?usage: pipeline.sh ROOT [SOURCE_TREE] [STAGE...]}" && cd "$1" && pwd); shift
src=$(cd -- "$here/../../.." && pwd)
if [[ $# -gt 0 && -d $1 ]]; then src=$(cd -- "$1" && pwd); shift; fi
stages=("$@"); [[ ${#stages[@]} -gt 0 ]] || stages=(sdks build size rebuild gate functional report)
: "${CAPSTONE_LLVM_BIN:?}" "${CAPSTONE_LLVM_BUILD_DIR:?}" "${CAPSTONE_QEMU_LOCK:?}"
kit=${KIT:?KIT: the QEMU kit (adapter/, images/, qemu-system-riscv64)}
args=${ARGS:-1 10}
export PYTHONDONTWRITEBYTECODE=1 JOBS=${JOBS:-12} CAPSTONE_REPO_ROOT=$here/../../..
unset CAPSTONE_CLANG CAPSTONE_MOVC_NULL_SCALAR CAPSTONE_QEMU_DOMAIN_LOCK
bpin=(taskset -c 16-23,48-55 nice -n 10 ionice -c3)
qpin=(taskset -c 24-31,56-63 nice -n 10)
gcc=${BENCH_NATIVE_CC:?BENCH_NATIVE_CC: buildroot riscv64 gcc (native references, walltime helper)}
export BENCH_NATIVE_CC
B=${BUILD:-$root/build}   # BUILD: build into another directory (never overwrite images in use)
run_qemu() {  # work dir, then run-qemu.py options
  local work=$1; shift
  rm -rf "$work"
  "${qpin[@]}" python3 "$here/qemu/run-qemu.py" --kit "$kit" --work "$work" --walltime "$root/walltime" \
    --image-timeout 900 --timeout 21600 "$@" > "$work.out" 2>&1 || true
  tail -1 "$work.out"
  grep -q '^completed=True missing=0$' "$work.out" || { echo "QEMU pass $work incomplete (see $work.out)" >&2; return 1; }
}
build_all() {
  rm -rf "$B"
  for arm in T1 T2; do
    "${bpin[@]}" "$here/build-all.sh" "$root/sdk-$arm/sdk" "$B/$arm"
    BENCH_MODE=count "${bpin[@]}" "$here/build-all.sh" "$root/sdk-$arm/sdk" "$B/$arm-count"
  done
  BENCH_MODE=native "${bpin[@]}" "$here/build-all.sh" - "$B/native"
  "${bpin[@]}" python3 "$here/build.py" --mode capstone --layout default --sdk "$root/sdk-T2/sdk" --out "$B/T2-default"
}
for stage in "${stages[@]}"; do
  echo "== $stage $(date +%T)"
  case $stage in
    sdks)
      "${bpin[@]}" bash "$here/build-sdks.sh" "$root" "${MUSL_CACHE:?}" "$src"
      "$gcc" -O2 -static "$here/qemu/walltime.c" -o "$root/walltime" ;;
    build|rebuild) build_all ;;
    size)
      run_qemu "$root/run-size" --set T1="$B/T1" --set T1c="$B/T1-count" --icount --args "1 5"
      python3 "$here/resize.py" --results "$root/run-size/results.json" --label T1 --count-label T1c ;;
    gate)
      python3 "$here/check-arms.py" "$B/T1" "$B/T2" --k-check --negative-control "$B/T2-default" \
        --tsv "$root/layout-gate.tsv" | tee "$root/layout-gate.txt" ;;
    functional)
      run_qemu "$root/run-functional" --set T1="$B/T1" --set T2="$B/T2" --native "$B/native" --icount --args "$args" \
        --nonreentrant "$here/nonreentrant.txt"
      run_qemu "$root/run-count" --set T1c="$B/T1-count" --set T2c="$B/T2-count" --args "1 5" ;;
    report)
      # every functional run (run-functional, then run-functional2.. re-runs of rebuilt programs,
      # which override per image), the counting run, and a console audit of each functional run
      r=(); con=()
      for f in "$root"/run-functional*/; do
        [[ -f $f/results.json ]] || continue
        r+=(--results "$f/results.json")
        python3 "$here/qemu/console-audit.py" "$f" --tsv "$f/console.tsv" > /dev/null
        con+=(--console "$f/console.tsv")
      done
      r+=(--results "$root/run-count/results.json")
      for arm in T1 T2; do
        python3 "$here/make-manifest.py" --arm $arm --build "$B/$arm" "${r[@]}" --count-label ${arm}c \
          --excluded "$B/$arm/excluded.tsv" --nonreentrant "$here/nonreentrant.txt" "${con[@]}" --out "$root/manifest-$arm.tsv" --sizing "$root/sizing-$arm.tsv"
      done
      python3 "$here/functional-table.py" "${r[@]}" --images "$B/T1/images.tsv" --out "$root/functional.tsv"
      python3 "$here/resize.py" "${r[@]:0:${#r[@]}-2}" --label T1 --check-only \
        > "$root/size-check.txt" || echo "size check: programs outside [0.6, 1.6] x target (see $root/size-check.txt)" ;;
    *) echo "unknown stage $stage" >&2; exit 2 ;;
  esac
done
}
main "$@"; exit
