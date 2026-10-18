#!/bin/bash
# C-81 VOID GATE input for the NARROW prototype, on the two sqlite CAUGHT-24 cells.
#
# Differs from twin-sqlite.sh in one way that matters: the narrow flag exists only in the
# compiler-lane toolchain, not in the frozen dev147 one, so BOTH arms are built with that
# toolchain and the only variable between them is the flag. The consequence is that the plain
# arm will NOT match repro322-silicon-2026-10-16/SHA256SUMS -- that record was produced by a
# different compiler revision. The record line is printed for information; for this experiment
# the control is "same toolchain, flag off", which is the comparison the void gate asks for.
#
# Compile-only: nothing here runs an image, so it takes no QEMU lock.
# Needs: C81_NARROW_BIN (a toolchain whose llc answers to -capstone-lower-narrow-libcall-c81),
# CAPSTONE_BUILDROOT_DIR (a checkout WITH submodules -- a worktree has none),
# C81_SCANNER_DIR, C81_OBJDUMP, C81_CLANG.
main() {
  local R; R=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 1
  local T=${C81_OUT:-/tmp/capstone/c81/sqlite-narrow} B=${C81_NARROW_BIN:?set C81_NARROW_BIN}
  local SC=$R/capstone/tests/rtl-smoke/repro322-silicon-2026-10-16
  local FLAG=-capstone-lower-narrow-libcall-c81
  "$B/llc" -mtriple=capstone64 --help-hidden 2>/dev/null | grep -q -- "$FLAG" \
    || { echo "GATE: $B does not carry $FLAG -- refusing to build"; return 2; }
  # build-corpus-images.sh reads its adapted input from $CAPSTONE_TMP_ROOT/sqlite-322-adapted, so
  # this must be the root that HOLDS it -- the same one twin-sqlite.sh uses. Pointing it at a fresh
  # directory makes both arms fail with "no adapted input", which is a missing input and not a result.
  export CAPSTONE_LLVM_BIN=$B CAPSTONE_TMP_ROOT=${CAPSTONE_TMP_ROOT:-/tmp/capstone/r322-1016/tmp}
  export CELLS="backupattach detachtrig"
  export CAPSTONE_BUILDROOT_DIR=${CAPSTONE_BUILDROOT_DIR:?set CAPSTONE_BUILDROOT_DIR}
  [ -f "$CAPSTONE_TMP_ROOT/sqlite-322-adapted/sqlite3.c" ] \
    || { echo "GATE: no adapted input at $CAPSTONE_TMP_ROOT/sqlite-322-adapted -- refusing"; return 2; }
  rm -rf "$T"; mkdir -p "$T"
  local bad=0
  nice -n 10 bash "$SC/build-corpus-images.sh" "$T/plain" > "$T/plain.log" 2>&1
  local rcp=$?; echo "plain rc=$rcp"; [ $rcp -eq 0 ] || bad=1
  EXTRA_MLLVM="-mllvm $FLAG" nice -n 10 bash "$SC/build-corpus-images.sh" "$T/narrow" > "$T/narrow.log" 2>&1
  local rcn=$?; echo "narrow rc=$rcn"; [ $rcn -eq 0 ] || bad=1
  [ $bad -eq 0 ] || { echo "a build FAILED -- the readings below would be void; tail of plain.log:"
                      tail -3 "$T/plain.log"; return 1; }
  for t in backupattach detachtrig; do
    local a=$T/plain/$t/r322-$t.dom b=$T/narrow/$t/r322-$t.dom
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "$t MISSING ($a / $b)"; bad=1; continue; fi
    echo "$t plain $(sha256sum "$a" | cut -c1-16) narrow $(sha256sum "$b" | cut -c1-16) \
record(other-compiler, informational) $(grep " r322-$t.dom" "$SC/SHA256SUMS" 2>/dev/null | cut -c1-16)"
    python3 "${C81_SCANNER_DIR:?set C81_SCANNER_DIR}/scan-memcpy-expansion.py" \
      --objdump "${C81_OBJDUMP:?}" --clang "${C81_CLANG:?}" --diff "$a" "$b" > "$T/diff-$t.txt" 2>&1
    echo "  diff rc=$? $(grep -o 'only in A: [0-9]*   only in B: [0-9]*' "$T/diff-$t.txt") $(grep -o 'A-only by unit: .*' "$T/diff-$t.txt")"
    echo "  CALLS-GAINED (void gate): $(python3 - "$a" "$b" "${C81_OBJDUMP}" <<'PY'
import subprocess,sys,re
def calls(p):
    d=subprocess.run([sys.argv[3],'-d',p],capture_output=True,text=True).stdout
    return len(re.findall(r'<(memcpy|memmove|memset)(\+0x[0-9a-f]+)?>', d))
print(calls(sys.argv[2])-calls(sys.argv[1]))
PY
)"
  done
  [ $bad -eq 0 ] && echo TWIN_SQLITE_NARROW_DONE || { echo "TWIN_SQLITE_NARROW_INCOMPLETE"; return 1; }
}
main "$@"; exit
