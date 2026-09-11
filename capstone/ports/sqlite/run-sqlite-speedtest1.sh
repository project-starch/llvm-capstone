#!/usr/bin/env bash
# NEAR-TWIN: run-speedtest1-measure.sh (I-7). This one drives the BRING-UP domain
# (speedtest1_domain.c -> build-sqlite-capstone.sh -> speedtest1_capstone.dom) with the testset
# compiled in. The twin drives the MEASUREMENT domain (speedtest1_measure.c ->
# build-sqlite-silicon.sh -> sqlite_silicon.dom) and emits cycle/instruction counts.
#
# speedtest1 in a Capstone domain on QEMU: SQLite's own benchmark with fixed arguments and the
# lookaside pool on, the allocator chain the paper measures. Output arrives on the hostcall
# payload the host prints when the domain returns; the domain ends with
# __CAPSTONE_SPEEDTEST1_DONE__ rc=<n>.
#
#   run-sqlite-speedtest1.sh            --memdb --size 1 --testset main --verify --stats
#   SPEEDTEST1_ARGS='"--memdb","--size","2"' run-sqlite-speedtest1.sh
#   SPEEDTEST1_SUBLET=1 run-sqlite-speedtest1.sh      the Sublet port of both allocators
#   SPEEDTEST1_HOOK=1 SPEEDTEST1_HOOK_SRC=<file.c> SQLITE_HOOK_PATCH=<file.patch> ...
#                                                     an instrument linked into the domain
#   SPEEDTEST1_PROBE=1 SPEEDTEST1_PROBE_SRC=<file.c> SPEEDTEST1_MARKER=<line> ...
#                                                     a probe instead of the benchmark
#   an experiment's runner sets these for its instrument and probes.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

SQLITE_VERSION=${SQLITE_VERSION:-3530300}
SQLITE_SRC_YEAR=${SQLITE_SRC_YEAR:-2026}
SRC_ROOT=$CAPSTONE_TMP_ROOT/sqlite-src
SPEEDTEST1_SRC=${SPEEDTEST1_SRC:-$SRC_ROOT/sqlite-src-$SQLITE_VERSION/test/speedtest1.c}
if [ ! -f "$SPEEDTEST1_SRC" ]; then
  # the amalgamation zip fetch-sqlite.sh takes does not ship speedtest1.c; the source tree does
  mkdir -p "$SRC_ROOT"
  curl -sfL -o "$SRC_ROOT/sqlite-src-$SQLITE_VERSION.zip" \
    "https://www.sqlite.org/$SQLITE_SRC_YEAR/sqlite-src-$SQLITE_VERSION.zip"
  unzip -qo "$SRC_ROOT/sqlite-src-$SQLITE_VERSION.zip" "sqlite-src-$SQLITE_VERSION/test/speedtest1.c" -d "$SRC_ROOT"
fi

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/speedtest1-build}
SHARE_DIR=${SHARE_DIR:-$OUT_DIR}
LOG_FILE=${LOG_FILE:-$CAPSTONE_TMP_ROOT/capstone-runtime-qemu-speedtest1.log}
REGION=${SPEEDTEST1_REGION_SIZE:-1048576}          # the payload, as the SLT runner sizes it
export SQLITE_OPT_LEVEL=${SQLITE_OPT_LEVEL:--O1}
export SQLITE_LOOKASIDE=${SQLITE_LOOKASIDE:-1200,40}
# SPEEDTEST1_STACK_ARENA=<bytes> carves memsys5's arena from the domain's stack region (2 MiB
# fits --size 1 with room for the stack); the in-image heap then only needs to exist.
STACK_ARENA=${SPEEDTEST1_STACK_ARENA:-1441792}
DOMAIN_ARGS="${SPEEDTEST1_ARGS:+-DSPEEDTEST1_ARGS=$SPEEDTEST1_ARGS} ${SPEEDTEST1_STOP_AT:+-DSPEEDTEST1_STOP_AT=$SPEEDTEST1_STOP_AT} -DSPEEDTEST1_STACK_ARENA=$STACK_ARENA"
# SPEEDTEST1_SUBLET=1: the Sublet port of memsys5 and lookaside (sublet/sublet-3530300.patch). The
# pool is then a region the host shares linear, SPEEDTEST1_ARENA bytes (--arena), and the tables
# region holds memsys5's tables. SPEEDTEST1_MARKER names what counts as success.
SUBLET=${SPEEDTEST1_SUBLET:-0}

# CAPSTONE_GP_NONLIN=1 IS REQUIRED, AND ITS ABSENCE LOOKS LIKE A PORT BUG RATHER THAN A MISSING
# EXPORT. QEMU re-fabricates gp from pc_cap at every call, and pc_cap is linear again once the first
# call returns to the entry frame -- so without this the fabricated gp is LINEAR, the compiler's
# `movc` of a live code capability MOVES it rather than copying, the source is nulled, and the next
# `cjalr` through it fails with "cs.cjalr requires capability in rs1". The fault lands deep in the
# run, post-entry, with no hint that the cause is an unset variable: two attempts died that way on
# 2026-09-11 before the cause was found in a1-sqlite-reuse/run.sh, which is the only place that
# exported it despite THIS script advertising SPEEDTEST1_SUBLET=1 at the top as a supported form.
#
# The knob keeps the fabricated gp NONLIN, which is the type the entry glue gives it on purpose.
# It lives in the mainline emulator (capstone-qemu op_helper.c, CAPSTONE_GP_NONLIN) as of the
# diag/domain-runs merge, so no separate "diagnostic QEMU" build is needed for it any more.
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}
# The allocators' memory comes from the host as regions. SPEEDTEST1_POOL is memsys5's heap in
# the unprotected arm, control bytes inside, so it holds POOL/65 atoms; the port's pool is
# those atoms times 64, so both arms carve the same blocks and every address matches. The
# tables region holds memsys5's tables under the port (57 bytes an atom with the handles) and
# the instrument's table (16 bytes an atom). A region above 4 MiB comes from the kernel's CMA
# area, hence cma= on the guest's command line. 1441792 is the size the pinned passes used
# (22181 atoms); --size 100 wants a pool of 136314880 (2^21 atoms, 130 MiB).
POOL=${SPEEDTEST1_POOL:-1441792}
ATOMS=$((POOL / 65))
ARENA=$((ATOMS * 64))
mib() { echo $(( ($1 + 1048575) / 1048576 * 1048576 )); }
HOST_ARGS="--tail --pool $POOL --tables $(mib $((ATOMS * 16 + 65536)))"
if [ "$SUBLET" = 1 ]; then
  export SQLITE_SUBLET_PATCH=${SQLITE_SUBLET_PATCH:-$SCRIPT_DIR/sublet/sublet-3530300.patch}
  DOMAIN_ARGS="$DOMAIN_ARGS -DSPEEDTEST1_SUBLET=1"
  HOST_ARGS="--tail --arena $ARENA --tables $(mib $((ATOMS * 57 + ATOMS * 16 + 131072)))"
fi
KERNEL_ARGS="--kernel-arg cma=${SPEEDTEST1_CMA:-1G}"
export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-8388608}
MARKER=${SPEEDTEST1_MARKER:-'__CAPSTONE_SPEEDTEST1_DONE__ rc=0'}
EXTRA_SRC=
# SPEEDTEST1_PROBE=n: a probe runs instead of the benchmark. SPEEDTEST1_PROBE_SRC is the source
# that defines speedtest1_probe, linked in beside speedtest1_domain.c; SPEEDTEST1_MARKER then
# names the probe's own line.
if [ -n "${SPEEDTEST1_PROBE:-}" ]; then
  : "${SPEEDTEST1_PROBE_SRC:?SPEEDTEST1_PROBE needs SPEEDTEST1_PROBE_SRC, the source of speedtest1_probe}"
  DOMAIN_ARGS="$DOMAIN_ARGS -DSPEEDTEST1_PROBE=$SPEEDTEST1_PROBE"
  EXTRA_SRC=$SPEEDTEST1_PROBE_SRC
fi
# SPEEDTEST1_HOOK=1: an instrument in the domain. SPEEDTEST1_HOOK_SRC is its source, linked in
# beside speedtest1_domain.c; SQLITE_HOOK_PATCH puts its calls into copies of the sources in the
# build directory, so speedtest1.c is copied there first and included from there. Off by default:
# the benchmark alone is the port's question, the instrument is an experiment's
# (its runner sets both).
HOOK=${SPEEDTEST1_HOOK:-0}
if [ "$HOOK" = 1 ]; then
  : "${SPEEDTEST1_HOOK_SRC:?SPEEDTEST1_HOOK=1 needs SPEEDTEST1_HOOK_SRC, the source of the instrument}"
  : "${SQLITE_HOOK_PATCH:?SPEEDTEST1_HOOK=1 needs SQLITE_HOOK_PATCH, the patch that calls it}"
  export SQLITE_HOOK_PATCH
  mkdir -p "$OUT_DIR"
  cp "$SPEEDTEST1_SRC" "$OUT_DIR/speedtest1.c"
  SPEEDTEST1_SRC=$OUT_DIR/speedtest1.c
  EXTRA_SRC="${EXTRA_SRC:+$EXTRA_SRC }$SPEEDTEST1_HOOK_SRC"
  DOMAIN_ARGS="$DOMAIN_ARGS -DSPEEDTEST1_HOOK=1 ${SPEEDTEST1_HOOK_FLAGS:-}"
fi
export DOMAIN_EXTRA_SRC="${DOMAIN_EXTRA_SRC:+$DOMAIN_EXTRA_SRC }$EXTRA_SRC"
export DOMAIN_EXTRA_FLAGS="-DSQLITE_HEAP_SIZE=${SPEEDTEST1_HEAP:-16384} -DSQLITE_HC_REGION_SIZE=$REGION -DSPEEDTEST1_SRC=\"$SPEEDTEST1_SRC\" $DOMAIN_ARGS ${DOMAIN_EXTRA_FLAGS:-}"
export HOST_EXTRA_DEFS="-DSQLITE_HC_REGION_SIZE=$REGION ${HOST_EXTRA_DEFS:-}"

mkdir -p "$OUT_DIR" "$SHARE_DIR"
rm -f "$SHARE_DIR/speedtest1_capstone.dom" "$SHARE_DIR/sqlite_host.user"
DOMAIN_SRC="$SCRIPT_DIR/speedtest1_domain.c" OUT_DIR="$OUT_DIR" OUT_DOM="$SHARE_DIR/speedtest1_capstone.dom" \
  bash "$SCRIPT_DIR/build-sqlite-capstone.sh"
OUT_DIR="$OUT_DIR" OUT_HOST="$SHARE_DIR/sqlite_host.user" \
  bash "$SCRIPT_DIR/build-sqlite-host.sh"

python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE_DIR" \
  --log-file "$LOG_FILE" \
  --timeout-multiplier "${SPEEDTEST1_TIMEOUT_MULTIPLIER:-40}" \
  $KERNEL_ARGS \
  --guest-command \
    "cp /mnt/host/sqlite_host.user /tmp/sqlite_host.user && chmod 0755 /tmp/sqlite_host.user && /tmp/sqlite_host.user /mnt/host/speedtest1_capstone.dom $HOST_ARGS" \
  --success-marker "$MARKER"

# SQLite's own statistics from the payload, as a file; an instrument's text is its experiment's
# to cut from the log
grep '^-- ' "$LOG_FILE" > "$OUT_DIR/speedtest1.rep1.stats.txt" || true
echo "stats: $OUT_DIR/speedtest1.rep1.stats.txt"
echo "run-sqlite-speedtest1.sh completed. Full serial log: $LOG_FILE"
