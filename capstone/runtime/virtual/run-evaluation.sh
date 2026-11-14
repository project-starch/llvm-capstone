#!/usr/bin/env bash
# run-evaluation.sh OUT: the evaluation on the trap route, once the kernel runs protected programs.
#
# Every step runs many programs per boot (trap_platform.py, bug-corpora/tools/virtualvm.py): a Sublet
# fault ends one program, a watchdog (CAPSTONE_STEP_TIMEOUT, default 600 s) kills a hung one, and only a
# dead guest costs a reboot, which the step records as a harness failure, never as a verdict.
#
# Steps, in order. A step whose inputs are unset is skipped and says so.
#   kit       KIT, or a new OUT/kit from SUBLET_QEMU, SUBLET_KERNEL, SUBLET_FIRMWARE (trap_platform.py)
#   gate      PROTECTED_FAULT (a .dom that makes a Sublet access fault) and PROTECTED_OK (a .dom that
#             exits 0) in one boot: the first must die by SIGSEGV with the launcher's fault line, the
#             second must then exit 0. The run stops here when the kernel does not pass.
#   juliet    JULIET_BUILD and JULIET_HELPER: bug-corpora/juliet/run.py
#   corpora   SDK and LLVM_BIN: bug-corpora/tools/run-virtual-cases.py for each of CORPORA x ARMS
#   plan      EVAL_PLAN: a file of further runner commands, one per line (# comments), run with KIT,
#             SDK, LLVM_BIN and OUT in the environment; the reuse study and the prebuilt nested arms
#             go here, because each needs its own build directory
#
# Environment for QEMU's lock: CAPSTONE_QEMU_LOCK (default ~/asr-work/qemu-gate.lock) and
# CAPSTONE_QEMU_DOMAIN_LOCK (default OUT/lock).
set -uo pipefail
OUT=${1:?usage: run-evaluation.sh OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CORPORA=${CORPORA:-"ffmpeg/plain-heap-repros ffmpeg/plain-temporal-repros ffmpeg/subobject-repros ffmpeg/carved-repros
memcached/plain-heap-repros memcached/plain-temporal-repros wireshark/plain-heap-repros wireshark/plain-temporal-repros"}
ARMS=${ARMS:-virtual-malloc}

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export CAPSTONE_QEMU_LOCK=${CAPSTONE_QEMU_LOCK:-$HOME/asr-work/qemu-gate.lock}
export CAPSTONE_QEMU_DOMAIN_LOCK=${CAPSTONE_QEMU_DOMAIN_LOCK:-$OUT/lock}
SUMMARY=$OUT/summary.tsv
printf 'step\trc\tseconds\n' > "$SUMMARY"

main() {
  step_kit
  step_gate || { echo "STOPPED: the kernel does not run protected programs with a fault line (see $OUT/gate)"; exit 1; }
  step_juliet
  step_corpora
  step_plan
  echo "DONE: $SUMMARY"
  cat "$SUMMARY"
}

step_kit() {
  if [ -n "${KIT:-}" ]; then
    echo "=== kit: $KIT (given)"
    return 0
  fi
  KIT=$OUT/kit
  [ -d "$KIT" ] || python3 "$HERE/trap_platform.py" make-kit "$KIT" || exit 1
  export KIT
  echo "=== kit: $KIT"
}

step_gate() {
  if [ -z "${PROTECTED_FAULT:-}" ] || [ -z "${PROTECTED_OK:-}" ]; then
    echo "=== gate skipped: set PROTECTED_FAULT and PROTECTED_OK"
    return 0
  fi
  local fault ok
  fault=$(basename "$PROTECTED_FAULT")
  ok=$(basename "$PROTECTED_OK")
  timed gate python3 "$HERE/trap_platform.py" run --kit "$KIT" --work "$OUT/gate" --step-timeout 120 \
    "$PROTECTED_FAULT" "$PROTECTED_OK" | tee "$OUT/gate.txt"
  grep -q "^$fault signal 11 | capstone-exec: domain fault cause=" "$OUT/gate.txt" &&
    grep -q "^$ok exit 0$" "$OUT/gate.txt"
}

step_juliet() {
  if [ -z "${JULIET_BUILD:-}" ] || [ -z "${JULIET_HELPER:-}" ]; then
    echo "=== juliet skipped: set JULIET_BUILD and JULIET_HELPER"
    return 0
  fi
  timed juliet python3 "$REPO/capstone/bug-corpora/juliet/run.py" --build "$JULIET_BUILD" --kit "$KIT" \
    --helper "$JULIET_HELPER" --out "$OUT/juliet"
}

step_corpora() {
  if [ -z "${SDK:-}" ] || [ -z "${LLVM_BIN:-}" ]; then
    echo "=== corpora skipped: set SDK and LLVM_BIN"
    return 0
  fi
  local corpus arm name
  for corpus in $CORPORA; do
    for arm in $ARMS; do
      name=$(echo "$corpus-$arm" | tr / -)
      timed "$name" python3 "$REPO/capstone/bug-corpora/tools/run-virtual-cases.py" \
        --corpus "$REPO/capstone/bug-corpora/$corpus" --arm "$arm" --virtual-kit "$KIT" --sdk "$SDK" \
        --llvm-bin "$LLVM_BIN" --out "$OUT/corpora/$name" --raw "$OUT/raw/$name"
    done
  done
}

step_plan() {
  if [ -z "${EVAL_PLAN:-}" ]; then
    echo "=== plan skipped: set EVAL_PLAN"
    return 0
  fi
  export OUT SDK LLVM_BIN
  local n=0 line
  while IFS= read -r line; do
    case $line in ''|'#'*) continue ;; esac
    n=$((n + 1))
    timed "plan-$n" bash -c "cd '$REPO' && $line"
  done < "$EVAL_PLAN"
}

# timed NAME COMMAND...: run COMMAND, log it to OUT/logs/NAME.log and record its status.
timed() {
  local name=$1 t0 rc
  shift
  mkdir -p "$OUT/logs"
  echo "=== $name"
  t0=$(date +%s)
  "$@" 2>&1 | tee "$OUT/logs/$name.log"
  rc=${PIPESTATUS[0]}
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$SUMMARY"
  echo "=== $name rc=$rc"
  return "$rc"
}

main
