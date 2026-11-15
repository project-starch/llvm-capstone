#!/usr/bin/env bash
# run-evaluation.sh OUT: the evaluation on the trap route, once the kernel runs protected programs.
#
# Every step runs many programs per boot (trap_platform.py, bug-corpora/tools/virtualvm.py): a Sublet
# fault ends one program, a watchdog (CAPSTONE_STEP_TIMEOUT, default 600 s) kills a hung one, and only a
# dead guest costs a reboot, which the step records as a harness failure, never as a verdict.
#
# EVAL_STEPS names the steps in order (default: gate juliet corpora plan). Besides those four it takes
# juliet:SET:PART with PART one of census, seeded, probe or judge, so a run can put the long spatial census
# last. A step whose inputs are unset is skipped and says so.
#
# Parallel VMs: Juliet batches run in JULIET_LANES lanes (run.py --lanes, default 16 here), the corpora
# and the plan lines each in their own lane, up to EVAL_LANES (default 16) at once. Each lane holds the
# global lock shared and its own domain lock CAPSTONE_QEMU_DOMAIN_LOCK.<lane>.
#   kit       KIT, or a new OUT/kit from SUBLET_QEMU, SUBLET_KERNEL, SUBLET_FIRMWARE (trap_platform.py)
#   gate      PROTECTED_FAULT (a .dom that makes a Sublet access fault) and PROTECTED_OK (a .dom that
#             exits 0) in one boot: the first must die by SIGSEGV with the launcher's fault line, the
#             second must then exit 0. The run stops here when the kernel does not pass.
#   juliet    JULIET_PREBUILT, JULIET_SRC, JULIET_HELPER, SDK and LLVM_BIN: for each set of JULIET_SETS
#             (default: temporal spatial), bug-corpora/juliet/run.py on the census build and the seeded
#             build, probe.py, and judge.py's first judgement. A set without a prebuilt census build
#             (spatial: 18,276 images, 2.9 GB) builds it into OUT first with build.py (about a minute).
#             The twin and ASan rounds of run-set.sh follow from the first judgement's verdicts.
#   corpora   SDK and LLVM_BIN: bug-corpora/tools/run-virtual-cases.py for each of CORPORA x ARMS, with the
#             words of CORPUS_ARGS_DIR/<corpus-with-dashes>-cc-args.txt when that file exists
#   plan      EVAL_PLAN: a file of further runner commands, one per line (# comments), run with KIT,
#             logged as plan<PLAN_TAG>-<n> (PLAN_TAG keeps the logs of a second plan apart),
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
export JULIET_LANES=${JULIET_LANES:-16}
# Each Juliet program's own limit (juliet-timeout). A program start costs one to three seconds under
# QEMU and 16 VMs at once slow each guest, so the 10 s of the deleted route cut faulting runs short
# (run-2 spatial seeded, 14 Nov 2026); 30 s still ends a looping case.
JULIET_LIMIT_MS=${JULIET_LIMIT_MS:-30000}
EVAL_LANES=${EVAL_LANES:-16}
SUMMARY=$OUT/summary.tsv
[ -f "$SUMMARY" ] || printf 'step\trc\tseconds\n' > "$SUMMARY"

main() {
  local step set part
  step_kit
  for step in ${EVAL_STEPS:-gate juliet corpora plan}; do
    case $step in
      gate) step_gate || { echo "STOPPED: the kernel does not run protected programs with a fault line (see $OUT/gate)"; exit 1; } ;;
      juliet) for set in ${JULIET_SETS:-temporal spatial}; do
                for part in census seeded probe judge; do juliet_part "$set" "$part"; done
              done ;;
      juliet:*:*) set=${step#juliet:}; part=${set#*:}; set=${set%%:*}; juliet_part "$set" "$part" ;;
      corpora) step_corpora ;;
      plan) step_plan ;;
      *) echo "STOPPED: unknown step $step"; exit 2 ;;
    esac
  done
  # Steps in which the kernel refused an access through the program's reference: counted apart.
  python3 "$HERE/kernel-faults.py" "$OUT" | tee "$OUT/kernel-faults.txt"
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

juliet_part() {
  local set=$1 part=$2
  if [ -z "${JULIET_PREBUILT:-}" ] || [ -z "${JULIET_SRC:-}" ] || [ -z "${JULIET_HELPER:-}" ] ||
     [ -z "${SDK:-}" ] || [ -z "${LLVM_BIN:-}" ]; then
    echo "=== juliet skipped: set JULIET_PREBUILT, JULIET_SRC, JULIET_HELPER, SDK and LLVM_BIN"
    return 0
  fi
  local J=$REPO/capstone/bug-corpora/juliet d=$JULIET_PREBUILT/$set out=$OUT/juliet/$set build
  mkdir -p "$out"
  build=$d/build
  [ -d "$build" ] || build=$out/build
  case $part in
    census)
      if [ ! -d "$build" ]; then
        timed "juliet-$set-build" python3 "$J/build.py" --manifest "$d/manifest.json" --juliet "$JULIET_SRC" \
          --sdk "$SDK" --out "$build"
      fi
      timed "juliet-$set-run" python3 "$J/run.py" --build "$build" --kit "$KIT" --helper "$JULIET_HELPER" \
        --limit-ms "$JULIET_LIMIT_MS" --out "$out/run" ;;
    seeded)
      timed "juliet-$set-run-seeded" python3 "$J/run.py" --build "$d/build-seeded" --kit "$KIT" \
        --helper "$JULIET_HELPER" --limit-ms "$JULIET_LIMIT_MS" --out "$out/run-seeded" ;;
    probe)
      timed "juliet-$set-probe" python3 "$J/probe.py" --sdk "$SDK" --kit "$KIT" --helper "$JULIET_HELPER" \
        --out "$out/probe" ;;
    judge)
      timed "juliet-$set-judge" python3 "$J/judge.py" --manifest "$d/manifest.json" --juliet "$JULIET_SRC" \
        --build "$build" --run "$out/run" --seeded-build "$d/build-seeded" --seeded-run "$out/run-seeded" \
        --llvm-bin "$LLVM_BIN" --out "$out/results" --probe "$out/probe/probe.json" ;;
    *) echo "STOPPED: unknown Juliet part $part"; exit 2 ;;
  esac
}

step_corpora() {
  if [ -z "${SDK:-}" ] || [ -z "${LLVM_BIN:-}" ]; then
    echo "=== corpora skipped: set SDK and LLVM_BIN"
    return 0
  fi
  local corpus arm name args
  for corpus in $CORPORA; do
    args=()
    if [ -n "${CORPUS_ARGS_DIR:-}" ] && [ -f "$CORPUS_ARGS_DIR/$(echo "$corpus" | tr / -)-cc-args.txt" ]; then
      read -r -a args < "$CORPUS_ARGS_DIR/$(echo "$corpus" | tr / -)-cc-args.txt"
    fi
    for arm in $ARMS; do
      name=$(echo "$corpus-$arm" | tr / -)
      in_lane timed "$name" python3 "$REPO/capstone/bug-corpora/tools/run-virtual-cases.py" \
        --corpus "$REPO/capstone/bug-corpora/$corpus" --arm "$arm" --virtual-kit "$KIT" --sdk "$SDK" \
        --llvm-bin "$LLVM_BIN" --out "$OUT/corpora/$name" --raw "$OUT/raw/$name" "${args[@]}"
    done
  done
  wait_lanes
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
    in_lane timed "plan${PLAN_TAG:-}-$n" bash -c "cd '$REPO' && $line"
  done < "$EVAL_PLAN"
  wait_lanes
}

# in_lane COMMAND...: run COMMAND in the background in the next free lane (at most EVAL_LANES at once),
# under its own domain lock, so its boot runs beside the others.
LANE=0
in_lane() {
  while [ "$(jobs -rp | wc -l)" -ge "$EVAL_LANES" ]; do sleep 2; done
  LANE=$((LANE + 1))
  CAPSTONE_QEMU_DOMAIN_LOCK=$CAPSTONE_QEMU_DOMAIN_LOCK.e$LANE "$@" > /dev/null &
}

wait_lanes() {
  wait
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
