#!/bin/bash
# C-81 exposure, the sqlite stock silicon arm's two CAUGHT-24 rows (02 backupattach, 05 detachtrig): a plain rebuild (must
# equal repro322-silicon-2026-10-16/SHA256SUMS) and a compile-only twin with -capstone-lower-memops-via-libcall, then
# scan-memcpy-expansion.py --diff per image. build-sqlite-silicon.sh is not LTO, and EXTRA_MLLVM goes into SILICON, which
# every C object of the link is compiled with, so the flag reaches codegen directly. Nothing here runs an image.
# Needs: DEV147_BIN (the frozen dev147 toolchain bin), CAPSTONE_BUILDROOT_DIR (a checkout WITH submodules: the
# domdata-budget gate reads it, and a worktree has none), C81_SCANNER_DIR (the compiler lane's C-81 folder),
# C81_OBJDUMP and C81_CLANG (an llvm-objdump, and a clang that accepts the scanner's self-test flags).
main() {
  local R; R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
  local T=${C81_OUT:-/tmp/capstone/c81/sqlite} B=${DEV147_BIN:?set DEV147_BIN}
  local SC=$R/capstone/tests/rtl-smoke/repro322-silicon-2026-10-16
  export CAPSTONE_LLVM_BIN=$B CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone/r322-1016/tmp} CELLS="backupattach detachtrig"
  export CAPSTONE_BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:?set CAPSTONE_BUILDROOT_DIR}
  rm -rf "$T"; mkdir -p "$T"
  nice -n 10 bash "$SC/build-corpus-images.sh" "$T/plain" > "$T/plain.log" 2>&1; echo "plain rc=$?"
  EXTRA_MLLVM="-mllvm -capstone-lower-memops-via-libcall" nice -n 10 bash "$SC/build-corpus-images.sh" "$T/libcall" > "$T/libcall.log" 2>&1
  echo "libcall rc=$?"
  for t in backupattach detachtrig; do
    local a=$T/plain/$t/r322-$t.dom b=$T/libcall/$t/r322-$t.dom
    echo "$t plain $(sha256sum "$a" | cut -c1-16) record $(grep " r322-$t.dom" "$SC/SHA256SUMS" | cut -c1-16) libcall $(sha256sum "$b" | cut -c1-16)"
    python3 "${C81_SCANNER_DIR:?set C81_SCANNER_DIR}/scan-memcpy-expansion.py" --objdump "${C81_OBJDUMP:?}" --clang "${C81_CLANG:?}" \
      --diff "$a" "$b" > "$T/diff-$t.txt" 2>&1
    echo "  diff rc=$? $(grep -o 'only in A: [0-9]*   only in B: [0-9]*' "$T/diff-$t.txt") $(grep -o 'A-only by unit: .*' "$T/diff-$t.txt")"
  done
  echo TWIN_SQLITE_DONE
}
main "$@"; exit
