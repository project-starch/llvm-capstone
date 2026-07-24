#!/usr/bin/env bash
#
# Nightly build + test orchestrator for the Capstone LLVM fork.
#
# One entry point that (1) incrementally rebuilds the Capstone clang, (2) runs
# the compiler lit tests, then (3) runs the QEMU runtime suites SERIALLY -- the
# suites share the rootfs.ext2 write lock, so they must never overlap -- and
# (4) writes a timestamped PASS/FAIL report under $CAPSTONE_TMP_ROOT (i.e.
# /tmp/capstone/, never the repo).
#
# Suites are classified by their EXIT CODE: every run-*.sh here uses
# `set -euo pipefail` and exits non-zero on failure, so exit 0 == PASS. (A suite
# that only prints results without asserting would report PASS on exit 0; the
# full per-suite log is captured either way for inspection.)
#
# Usage:
#   run-nightly.sh [options]
#     --clean            also rebuild capstone-qemu + buildroot from their build
#                        scripts (slow); default is an incremental clang build
#     --skip-build       run suites against the current toolchain, no rebuild
#     --extended         also run the setup-heavier suites (sqlite/hostcall/nullblk)
#     --only a,b,c       run only the named suites (see --list); still serial
#     --list             print the suite names and exit
#     --timeout SECONDS  per-suite timeout (default 1800)
#     -j N               build parallelism (default ~70% of nproc)
#     -h | --help        this help
#
# Exit status: 0 iff the build and every executed suite passed.

set -uo pipefail  # NOT -e: we run every suite and report all outcomes.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/capstone-test-env.sh"

RUNTIME_DIR="$SCRIPT_DIR/runtime-qemu"
LIT="$CAPSTONE_LLVM_BUILD_DIR/bin/llvm-lit"

# ---- suite registry: "name|command[|timeout_seconds]" (cmd eval'd from repo root)
# Core tier: confirmed headless, self-contained, broadly validating. The three
# benchmark suites (coremark/rv8/beebs) are the primary backend regression gate;
# run-all-beebs boots 82 domains serially, so it carries its own long timeout.
BENCH_DIR="$CAPSTONE_REPO_ROOT/capstone/benchmarks"
CORE_SUITES=(
  "authority|bash $SCRIPT_DIR/capstone-authority/run-authority-suite.sh"
  "smoke|bash $RUNTIME_DIR/run-smoke.sh"
  "coremark|bash $RUNTIME_DIR/run-coremark.sh"
  "rv8|bash $BENCH_DIR/rv8/run-all-rv8.sh|3600"
  "beebs|bash $BENCH_DIR/beebs/run-all-beebs.sh|10800"
  "revoke-matrix|bash $RUNTIME_DIR/run-revoke-matrix-probe.sh"
  "revoke-on-free|bash $RUNTIME_DIR/run-revoke-on-free-probe.sh"
  "borrow-cost|bash $RUNTIME_DIR/run-borrow-cost-probe.sh"
  "tree-cost-O2|DOMAIN_OPT_LEVEL=-O2 bash $RUNTIME_DIR/run-tree-cost-probe.sh"
  "static-cap-globals|bash $RUNTIME_DIR/run-static-cap-globals-probe.sh"
  "intra-domain-mrev|bash $RUNTIME_DIR/run-intra-domain-mrev-revoke-probe.sh"
  "hier-revoke|bash $RUNTIME_DIR/run-hier-revoke-probe.sh"
  "shared-region|bash $RUNTIME_DIR/run-shared-region-probe.sh"
  "linear-uninit-corpus|bash $RUNTIME_DIR/run-linear-uninit-corpus-probe.sh"
)
# Extended tier: need kernel modules / extra setup; opt-in via --extended.
EXTENDED_SUITES=(
  "sqlite-borrow-revoke|bash $RUNTIME_DIR/run-sqlite-borrow-revoke-probe.sh"
  "sqlite-hier-revoke|bash $RUNTIME_DIR/run-sqlite-hier-revoke-probe.sh"
  "sqlite-sealed-callback|bash $RUNTIME_DIR/run-sqlite-sealed-callback-revoke-probe.sh"
  "hostcall-all|bash $RUNTIME_DIR/run-hostcall-all.sh"
  "nullblk-all|bash $RUNTIME_DIR/run-nullblk-all.sh"
)

# ---- options ------------------------------------------------------------------
DO_CLEAN=0 SKIP_BUILD=0 EXTENDED=0 ONLY="" TIMEOUT=1800
JOBS=${JOBS:-$(( $(nproc) * 7 / 10 ))}; [ "$JOBS" -lt 1 ] && JOBS=1

print_list() {
  echo "core:"; printf '  %s\n' "${CORE_SUITES[@]%%|*}"
  echo "extended (--extended):"; printf '  %s\n' "${EXTENDED_SUITES[@]%%|*}"
}
while [ $# -gt 0 ]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --extended) EXTENDED=1 ;;
    --only) ONLY="$2"; shift ;;
    --list) print_list; exit 0 ;;
    --timeout) TIMEOUT="$2"; shift ;;
    -j) JOBS="$2"; shift ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SUITES=("${CORE_SUITES[@]}")
[ "$EXTENDED" -eq 1 ] && SUITES+=("${EXTENDED_SUITES[@]}")

# ---- output dir + report ------------------------------------------------------
TS=$(date +%Y%m%d_%H%M%S)
OUT="$CAPSTONE_TMP_ROOT/nightly-$TS"
mkdir -p "$OUT"
REPORT="$OUT/report.md"
declare -A RESULT DURATION
OVERALL_OK=1

log() { printf '%s\n' "$*" | tee -a "$OUT/console.log"; }
sha() { git -C "$CAPSTONE_REPO_ROOT" rev-parse --short "$1" 2>/dev/null || echo "?"; }

log "== Capstone nightly $TS =="
log "output   : $OUT"
log "parent   : $(sha HEAD) ($(git -C "$CAPSTONE_REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
log "jobs=$JOBS timeout=${TIMEOUT}s clean=$DO_CLEAN skip_build=$SKIP_BUILD extended=$EXTENDED"

# ---- stage 1: build -----------------------------------------------------------
BUILD_OK=1
if [ "$SKIP_BUILD" -eq 1 ]; then
  log "[build] skipped (--skip-build)"
else
  log "[build] clang (incremental, -j$JOBS) ..."
  if ! cmake --build "$CAPSTONE_LLVM_BUILD_DIR" --target clang -j"$JOBS" \
        >"$OUT/build-clang.log" 2>&1; then
    BUILD_OK=0; OVERALL_OK=0; log "[build] clang FAILED (see build-clang.log)"
  fi
  # QEMU staleness guard: a default (non---clean) nightly does NOT rebuild
  # qemu-system-riscv64, so a submodule source move (e.g. a new instrumentation op)
  # can leave the shared binary behind its own pinned source -- the op then decodes
  # as an illegal instruction in-domain. Cheap fix: if the binary is older than any
  # tracked qemu source, do an incremental ninja relink (seconds), not a --clean build.
  if [ "$BUILD_OK" -eq 1 ] && [ "$DO_CLEAN" -eq 0 ]; then
    QBUILD="$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/build"
    QBIN="$QBUILD/qemu-system-riscv64"
    QSRC="$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/target"
    if [ -f "$QBIN" ] && [ -f "$QBUILD/build.ninja" ] && \
       [ -n "$(find "$QSRC" -type f -newer "$QBIN" -print -quit 2>/dev/null)" ]; then
      log "[build] qemu binary older than its source -> incremental ninja relink ..."
      if ninja -C "$QBUILD" qemu-system-riscv64 \
            >"$OUT/build-qemu-incremental.log" 2>&1; then
        log "[build] qemu incremental relink ok"
      else
        BUILD_OK=0; OVERALL_OK=0; log "[build] qemu incremental relink FAILED"
      fi
    fi
  fi
  if [ "$DO_CLEAN" -eq 1 ] && [ "$BUILD_OK" -eq 1 ]; then
    for bs in "$CAPSTONE_REPO_ROOT/capstone/capstone-qemu/build-qemu.sh" \
              "$RUNTIME_DIR/build-buildroot.sh"; do
      [ -x "$bs" ] || continue
      log "[build] --clean: $bs ..."
      bash "$bs" >"$OUT/build-$(basename "$bs" .sh).log" 2>&1 || {
        BUILD_OK=0; OVERALL_OK=0; log "[build] $(basename "$bs") FAILED"; }
    done
  fi
  [ "$BUILD_OK" -eq 1 ] && log "[build] ok"
fi

# A broken toolchain makes every suite result meaningless -> stop here.
if [ "$BUILD_OK" -eq 0 ]; then
  log "[abort] build failed; skipping suites"
fi

run_one() { # $1=name  $2=command  $3=timeout(optional, default $TIMEOUT)
  local name="$1" cmd="$2" to="${3:-$TIMEOUT}" start rc dur
  local suitelog="$OUT/$name.log"
  [ -n "$to" ] || to="$TIMEOUT"
  log "[suite] $name (timeout ${to}s) ..."
  start=$SECONDS
  ( cd "$CAPSTONE_REPO_ROOT" && timeout "$to" bash -c "$cmd" ) \
      >"$suitelog" 2>&1
  rc=$?
  dur=$(( SECONDS - start ))
  DURATION[$name]=$dur
  if [ $rc -eq 0 ]; then RESULT[$name]=PASS
  elif [ $rc -eq 124 ]; then RESULT[$name]=TIMEOUT; OVERALL_OK=0
  else RESULT[$name]="FAIL($rc)"; OVERALL_OK=0
  fi
  log "        -> ${RESULT[$name]} (${dur}s)"
}

# ---- stage 2: lit (fast, parallel-safe) ---------------------------------------
if [ "$BUILD_OK" -eq 1 ]; then
  CLANG_CAP_TESTS=$(ls "$CAPSTONE_REPO_ROOT"/clang/test/CodeGen/*capstone*.c \
                       "$CAPSTONE_REPO_ROOT"/clang/test/CodeGen/*capstone*.cc \
                       "$CAPSTONE_REPO_ROOT"/clang/test/CodeGen/builtins-capstone.c \
                    2>/dev/null | sort -u | tr '\n' ' ')
  run_one "lit" "$LIT llvm/test/CodeGen/Capstone/ $CLANG_CAP_TESTS"

  # ---- stage 3: QEMU suites (SERIAL, rootfs lock guarded) ---------------------
  # flock guards against a second nightly; a manual concurrent QEMU run must
  # still be avoided by convention (announce in agent-handoff/COORDINATION.md).
  LOCK="$CAPSTONE_TMP_ROOT/nightly-qemu.lock"; : >"$LOCK" 2>/dev/null || true
  exec 9>"$LOCK"
  flock 9 || log "[warn] could not take QEMU lock; proceeding serially anyway"
  for entry in "${SUITES[@]}"; do
    IFS='|' read -r name cmd stimeout <<< "$entry"
    if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
    run_one "$name" "$cmd" "$stimeout"
  done
  flock -u 9 2>/dev/null || true
fi

# ---- stage 4: report ----------------------------------------------------------
{
  echo "# Capstone nightly report -- $TS"
  echo
  echo "- parent: \`$(sha HEAD)\` ($(git -C "$CAPSTONE_REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null))"
  echo "- build: $([ "$SKIP_BUILD" -eq 1 ] && echo skipped || ([ "$BUILD_OK" -eq 1 ] && echo ok || echo FAILED))"
  echo "- output: \`$OUT\`"
  echo
  echo "| suite | result | duration | log |"
  echo "|---|---|---|---|"
  for entry in "lit|" "${SUITES[@]}"; do
    name="${entry%%|*}"
    [ -n "${RESULT[$name]:-}" ] || continue
    echo "| $name | ${RESULT[$name]} | ${DURATION[$name]}s | \`$name.log\` |"
  done
  echo
  echo "Overall: $([ "$OVERALL_OK" -eq 1 ] && echo '**PASS**' || echo '**FAIL**')"
} > "$REPORT"

log ""
log "== report =="
cat "$REPORT" | tee -a "$OUT/console.log" >/dev/null
sed -n '/| suite |/,$p' "$REPORT"
log ""
log "full report: $REPORT"
[ "$OVERALL_OK" -eq 1 ] && exit 0 || exit 1
