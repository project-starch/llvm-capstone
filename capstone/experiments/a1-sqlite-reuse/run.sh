#!/usr/bin/env bash
# A1 in a Capstone domain: does level 0 see the objects in the custom allocator? SQLite's
# lookaside pool over memsys5, speedtest1 as the workload, every allocation and free at both
# levels recorded from inside the domain by memhook.c, on the unprotected build and on the
# Sublet port of both allocators. This script owns what is the experiment's: the instrument and
# its patches, the probes, the sizes, and the files a pass leaves. The build and the domain are
# the port's (capstone/ports/sqlite/run-sqlite-speedtest1.sh).
#
#   run.sh memsys5|sublet [--size N] [--reps R] [--pool BYTES] [--out DIR] [--ref DIR]
#       one arm of the benchmark, R repetitions (default 1). Each leaves <arm>.rep<i>.hook.txt
#       and .stats.txt in DIR, default results/<stamp>/, beside the images' sha256 and a
#       provenance record. Repetitions must be bit-identical. --ref names a recorded pass: the
#       verdict is then whether this arm's hook text is bit-identical to its rep1.
#   run.sh all [--size N ...]        memsys5, then sublet, into one DIR
#   run.sh probe N [--out DIR]       N in 1..6: the unprotected build first, as the control, then
#                                    the port; the verdict is against what probes.c says for N. A
#                                    control that fails ends the run with exit 75 and no verdict.
#
# Every domain run needs the diagnostic QEMU (capstone-qemu branch diag/domain-runs, built to
# CAPSTONE_QEMU_BINARY) with CAPSTONE_GP_NONLIN=1: without it the compiler's movc of a live code
# capability nulls its source at the first return to the entry frame. The script exports it.
# Either build of it runs every arm: the port writes a block through before its init, so a
# revoke that leaves the cursor at the base (Q-07) and one that leaves it at the end (the build
# the passes used) both give the same run -- see the README, what the numbers are scoped to.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd -- "$HERE/../../ports/sqlite" && pwd)
source "$HERE/../../tests/capstone-test-env.sh"

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 1 ] || usage
MODE=$1; shift
PROBE_N=
if [ "$MODE" = probe ]; then PROBE_N=${1:?probe needs a number, 1 to 6}; shift; fi
SIZE=1; REPS=1; POOL=; OUT=; REF=
while [ $# -gt 0 ]; do
  case $1 in
    --size) SIZE=$2; shift 2;;
    --reps) REPS=$2; shift 2;;
    --pool) POOL=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --ref) REF=$2; shift 2;;
    *) echo "unknown option $1" >&2; usage;;
  esac
done
# The pool is memsys5's heap in the unprotected arm, control bytes inside, so it holds POOL/65
# atoms; the port's pool is the same atoms times 64 (the port runner does that arithmetic).
# 1441792 is what the passes at --size 1 used, 22181 atoms; --size 100 wants 2^21 atoms,
# 136314880 bytes, 130 MiB; any other size is the caller's to size.
if [ -z "$POOL" ]; then
  case $SIZE in 1) POOL=1441792;; 100) POOL=136314880;; *) echo "--size $SIZE needs --pool" >&2; exit 2;; esac
fi
[ -n "$OUT" ] || OUT=$HERE/results/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
LOG_DIR=$CAPSTONE_TMP_ROOT/a1-sqlite-reuse
mkdir -p "$LOG_DIR"
export OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-build}
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}
export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-8388608}
ARGS="\"--memdb\",\"--size\",\"$SIZE\",\"--testset\",\"main\",\"--verify\",\"--stats\""
HALT='domain halted by capability fault'

provenance() {  # what produced the files beside it: enough to rebuild the images and check them
  local f=$OUT/provenance.txt qemu=$CAPSTONE_QEMU_BINARY qdir=
  [ -f "$f" ] && return 0
  qdir=$(cd -- "$(dirname -- "$qemu")" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null || true)
  {
    echo "date            $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "platform        Capstone domain on QEMU virt-capstone, purecap image (capstone64-unknown-elf), freestanding"
    echo "sqlite          ${SQLITE_VERSION:-3530300}, the port in capstone/ports/sqlite"
    echo "speedtest1      --memdb --size $SIZE --testset main --verify --stats, lookaside ${SQLITE_LOOKASIDE:-1200,40}"
    echo "pool            $POOL bytes in the unprotected arm ($((POOL / 65)) atoms of 64 bytes); the port's pool the same atoms, linear"
    echo "regions         sqlite_host.user --pool / --arena and --tables from the host, from the kernel's CMA area above 4 MiB (cma=${SPEEDTEST1_CMA:-1G}), released after the run"
    echo "llvm-capstone   $(git -C "$HERE" rev-parse --abbrev-ref HEAD) $(git -C "$HERE" rev-parse --short=12 HEAD)$(git -C "$HERE" diff --quiet -- capstone/ports/sqlite capstone/experiments || echo ' (with uncommitted changes)')"
    echo "buildroot       $(git -C "$CAPSTONE_BUILDROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true) $(git -C "$CAPSTONE_BUILDROOT_DIR" rev-parse --short=12 HEAD 2>/dev/null || true)"
    echo "qemu            $qemu${qdir:+; $(git -C "$qdir" rev-parse --abbrev-ref HEAD) $(git -C "$qdir" rev-parse --short=12 HEAD)}; CAPSTONE_GP_NONLIN=$CAPSTONE_GP_NONLIN CAPSTONE_REV_NODES=$CAPSTONE_REV_NODES"
    echo "port            $(sha256sum "$PORT/sublet/sublet-3530300.patch" | cut -c1-64)  capstone/ports/sqlite/sublet/sublet-3530300.patch"
    echo "primitives      $(sha256sum "$HERE/../../sublet/sublet.h" | cut -c1-64)  capstone/sublet/sublet.h"
    echo "instrument      $(sha256sum "$HERE/memhook.c" | cut -c1-64)  memhook.c, called from hook-3530300.patch (memsys5 arm) and hook-3530300-sublet.patch (sublet arm)"
  } > "$f"
}

run_port() {  # run_port <label> VAR=value... : the port runner, logging under this experiment's name
  local label=$1; shift
  export LOG_FILE=$LOG_DIR/$label.log
  env "$@" bash "$PORT/run-sqlite-speedtest1.sh"
}

run_arm() {  # run_arm memsys5|sublet
  local arm=$1 sublet=0 patch=$HERE/hook-3530300.patch rep
  if [ "$arm" = sublet ]; then sublet=1; patch=$HERE/hook-3530300-sublet.patch; fi
  provenance
  for rep in $(seq 1 "$REPS"); do
    echo "== $arm rep$rep: --size $SIZE, pool $POOL bytes"
    run_port "$arm.rep$rep" SPEEDTEST1_SUBLET=$sublet SPEEDTEST1_POOL=$POOL SPEEDTEST1_ARGS="$ARGS" \
      SPEEDTEST1_HOOK=1 SPEEDTEST1_HOOK_SRC="$HERE/memhook.c" SPEEDTEST1_HOOK_FLAGS=-DMEMHOOK_FREESTANDING SQLITE_HOOK_PATCH="$patch"
    awk '/^memhook: /{p=1} /^__CAPSTONE_SPEEDTEST1_DONE__/{p=0} p' "$LOG_FILE" > "$OUT/$arm.rep$rep.hook.txt"
    { grep '^-- ' "$LOG_FILE" || true; grep '^sublet: ' "$LOG_FILE" || true; } > "$OUT/$arm.rep$rep.stats.txt"
    if [ "$rep" = 1 ]; then
      sha256sum "${SHARE_DIR:-$OUT_DIR}/speedtest1_capstone.dom" | sed "s|  .*|  $arm.dom|" >> "$OUT/image.sha256"
    fi
    echo "   $(wc -l < "$OUT/$arm.rep$rep.hook.txt") hook lines, $(wc -l < "$OUT/$arm.rep$rep.stats.txt") stats lines"
    if [ "$rep" -gt 1 ]; then
      if cmp -s "$OUT/$arm.rep1.hook.txt" "$OUT/$arm.rep$rep.hook.txt"; then echo "   rep$rep bit-identical to rep1"
      else echo "VERDICT $arm: rep$rep DIFFERS from rep1"; return 1; fi
    fi
  done
  if [ -n "$REF" ]; then
    if cmp -s "$OUT/$arm.rep1.hook.txt" "$REF/$arm.rep1.hook.txt"; then echo "VERDICT $arm: bit-identical to $REF"
    else echo "VERDICT $arm: DIFFERS from $REF"; return 1; fi
  fi
}

control_marker() {  # what a probe prints when its read goes through
  case $1 in
    1) echo '__CAPSTONE_SPEEDTEST1_UAF_NOTRAP__ lookaside';;
    2) echo '__CAPSTONE_SPEEDTEST1_UAF_NOTRAP__ memsys5';;
    3) echo '__CAPSTONE_SPEEDTEST1_SIBLING__ memsys5';;
    4) echo '__CAPSTONE_SPEEDTEST1_SIBLING__ lookaside';;
    5) echo '__CAPSTONE_SPEEDTEST1_BOUNDS__ p[127] read';;
    6) echo '__CAPSTONE_SPEEDTEST1_BOUNDS_NOTRAP__ p[128] read';;
    *) echo "probe $1: there are six" >&2; exit 2;;
  esac
}
port_expects() { case $1 in 1|2|6) echo halt;; *) echo reads;; esac; }  # what the port does with it

run_probe() {  # run_probe N
  local n=$1 marker want out rc=0
  out=$OUT/probe$n.txt
  marker=$(control_marker "$n"); want=$(port_expects "$n")
  provenance
  echo "== probe $n, the control: the unprotected build"
  if ! run_port "probe$n.control" SPEEDTEST1_PROBE="$n" SPEEDTEST1_PROBE_SRC="$HERE/probes.c" \
       SPEEDTEST1_POOL=$POOL SPEEDTEST1_MARKER="$marker"; then
    echo "control failed: no '$marker' in $LOG_FILE; no verdict" | tee "$out"; exit 75
  fi
  { echo "probe $n"; echo "control   $(grep -m1 -F "$marker" "$LOG_FILE")"; } > "$out"
  echo "== probe $n, the port"
  if [ "$want" = halt ]; then
    # a halted domain stops QEMU before the shell prompt, so the port runner exits non-zero
    # either way; the fault line in the log is the result
    run_port "probe$n.sublet" SPEEDTEST1_SUBLET=1 SPEEDTEST1_PROBE="$n" SPEEDTEST1_PROBE_SRC="$HERE/probes.c" \
      SPEEDTEST1_POOL=$POOL SPEEDTEST1_MARKER="$HALT" || rc=$?
    if grep -q -F "$HALT" "$LOG_FILE"; then
      echo "port      $(grep -m1 -F "$HALT" "$LOG_FILE")" >> "$out"
      echo "VERDICT probe $n: PASS, the port halts the read; the control read through" | tee -a "$out"
    else
      echo "port      no fault line; $(grep -m1 -F "$marker" "$LOG_FILE" || echo 'no marker either')" >> "$out"
      echo "VERDICT probe $n: FAIL, the port did not halt" | tee -a "$out"; return 1
    fi
  else
    run_port "probe$n.sublet" SPEEDTEST1_SUBLET=1 SPEEDTEST1_PROBE="$n" SPEEDTEST1_PROBE_SRC="$HERE/probes.c" \
      SPEEDTEST1_POOL=$POOL SPEEDTEST1_MARKER="$marker" || rc=$?
    if [ "$rc" = 0 ]; then
      echo "port      $(grep -m1 -F "$marker" "$LOG_FILE")" >> "$out"
      echo "VERDICT probe $n: PASS, both read on; the port reaches one object and not its neighbour" | tee -a "$out"
    else
      echo "port      $(grep -m1 -F "$HALT" "$LOG_FILE" || echo 'no marker, no fault line')" >> "$out"
      echo "VERDICT probe $n: FAIL, the port did not read on" | tee -a "$out"; return 1
    fi
  fi
}

case $MODE in
  memsys5|sublet) run_arm "$MODE";;
  all) run_arm memsys5; run_arm sublet;;
  probe) run_probe "$PROBE_N";;
  *) usage;;
esac
echo "files: $OUT"
