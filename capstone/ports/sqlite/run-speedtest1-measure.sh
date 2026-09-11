#!/usr/bin/env bash
# Run SQLite's own speedtest1 inside the silicon-config capability domain, under QEMU.
#
# NEAR-TWIN: run-sqlite-speedtest1.sh (I-7). This one drives the MEASUREMENT domain
# (speedtest1_measure.c -> build-sqlite-silicon.sh -> sqlite_silicon.dom) and takes its testset at
# run time. The twin drives the BRING-UP domain (speedtest1_domain.c -> build-sqlite-capstone.sh
# -> speedtest1_capstone.dom) with the testset compiled in. Different tools, similar names.
#
# Shaped like run-sqlite-slt.sh and for the same reason: ONE assignment sets the region size for
# both builds, because host and domain are separate compilations of one #define and a drift between
# them is silent. The domain also refuses to run on a mismatch, but that gate is the backstop.
#
# WHAT THIS SCRIPT IS NOT. It is not the board. Board execution goes to the board lane; this is the
# emulated arm that has to be green before a boot is spent.
#
# THE TWO CONSTRAINTS THAT DECIDE THE INVOCATION, both measured 2026-09-10:
#
#  1. --testset MUST be named. The default is mix1, which contains json (SQLITE_OMIT_JSON), rtree
#     (not enabled), cte and star (decimal literals the tokenizer rejects under
#     SQLITE_OMIT_FLOATING_POINT) and fp (needs round()). Any of them reaches fatal_error. main, orm
#     and parsenumber run clean. The host refuses an invocation without --testset.
#
#  2. --size 1 IS THE CEILING for main and orm, on MEMORY, not on run time. The smallest memsys5
#     arena in which each completes, measured natively through the same
#     sqlite3_config(SQLITE_CONFIG_HEAP,...) path:
#
#         testset      size 1     size 5     size 20
#         main         1.5 MiB    6 MiB      32 MiB
#         orm          2 MiB      8 MiB      32 MiB
#         parsenumber  <256 KiB   <256 KiB   <256 KiB
#
#     A domain's dom_data cannot exceed 4,194,304 bytes (MAX_ALLOC_ORDER is 10, and the region is
#     one __get_free_pages call), and the heap is carved from it alongside the globals blob, the cap
#     table and the stack. So the arena ceiling is a little under 3 MiB. parsenumber is the only
#     testset whose size knob is free.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# 64 KiB, not the SLT path's 1 MiB: the input is a short command line and the output is at most a
# few kilobytes of report, and the region is charged to the host's mapping either way. Both halves
# take it from here.
source "$SCRIPT_DIR/speedtest1-geometry.sh"
REGION=$SPEEDTEST1_GEOM_REGION
export SQLITE_HC_REGION_SIZE=$REGION

# 1.75 MiB is THE LARGEST HEAP THAT FITS UNDER THE UNCHANGED 2 MiB STACK DECLARATION, and it is
# above main --size 1's measured 1.5 MiB minimum. Both halves of that sentence are load-bearing:
#
#  - larger does not work. 2.5 MiB built and passed domdata-budget.py and then faulted at
#    SQ: E/share1 under QEMU, before the domain was ever entered (arm B, 2026-09-10). The budget
#    tool's ceiling is necessary and NOT sufficient; do not raise this on the strength of it.
#  - the stack declaration does not need to move. Reducing it to 1 MiB was tried and is not what
#    broke arm B (arm E ran clean at 1 MiB), but it buys nothing here and changes the geometry every
#    other SQLite result was measured at.
#
# orm --size 1 needs a 2 MiB arena, which does NOT fit under a 2 MiB stack declaration -- but it runs
# clean at heap 2 MiB with the declaration reduced to 1 MiB. (A cycle figure once quoted here came
# from the host-timestamp set and was retracted with it; the geometry finding did not depend on it.)
# Since one SQLite image carries one compiled-in heap and only one may be staged per boot, that is
# the geometry both main and orm use, and it is now the shared default in speedtest1-geometry.sh
# rather than a number repeated in three files that disagreed.
export SQLITE_HEAP_SIZE=$SPEEDTEST1_GEOM_HEAP
export SQLITE_SILICON_STACK=$SPEEDTEST1_GEOM_STACK

export SPEEDTEST1_STUB_CLOCK=${SPEEDTEST1_STUB_CLOCK:-0}
export SQLITE_SPEEDTEST1_SRC=${SQLITE_SPEEDTEST1_SRC:-$(bash "$SCRIPT_DIR/fetch-sqlite-src.sh")/test/speedtest1.c}
[[ -f "$SQLITE_SPEEDTEST1_SRC" ]] || { echo "ERROR: no speedtest1.c at $SQLITE_SPEEDTEST1_SRC" >&2; exit 1; }

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-speedtest1}
export OUT_DIR

# --verify BY DEFAULT. The hash is the verdict for main and orm, and until now no committed
# invocation produced one: every hash quoted in the plan came from an ad-hoc run.
SPEEDTEST1_ARGS=${SPEEDTEST1_ARGS:---testset main --size 1 --verify}
case "$SPEEDTEST1_ARGS" in
  *--testset*) : ;;
  *) echo "ERROR: SPEEDTEST1_ARGS must name --testset (the default mix1 cannot run here)" >&2
     exit 1 ;;
esac

# SPEEDTEST1_ALLOCSTATS=1 turns on the domain's allocation census, which reports the SUM of every
# allocation the run makes alongside the peak. That sum is what a never-coalescing allocator such as
# revoke-on-free must hold, and it is how the revoke-on-free arm was scoped to one testset. It was
# reachable only by setting DOMAIN_EXTRA_DEFS by hand, so the table it produced could not be
# reproduced from the repo. It PERTURBS the measurement by a few per cent, so its runs are
# diagnostics and the reported cycle numbers come from builds without it.
# SPEEDTEST1_INSTRET=1 adds the domain-side minstret bracket. It is a SEPARATE IMAGE by design:
# adding minstret instrumentation to a domain has flipped a rung to a deterministic miscompute on
# this silicon, so the arms whose hashes are already verified must not carry it. Stage this one LAST.
if [[ "${SPEEDTEST1_INSTRET:-0}" == "1" ]]; then
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_SPEEDTEST1_INSTRET=1"
  echo "== minstret bracket ON -- separate image, stage it LAST (see the note in speedtest1_measure.c)"
fi
if [[ "${SPEEDTEST1_ALLOCSTATS:-0}" == "1" ]]; then
  DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DCAPSTONE_SPEEDTEST1_ALLOCSTATS=1"
  echo "== allocation census ON -- this run is a diagnostic, not a timing measurement"
fi
DOMAIN_EXTRA_DEFS="${DOMAIN_EXTRA_DEFS:-} -DSQLITE_HC_REGION_SIZE=${REGION}UL"
HOST_EXTRA_DEFS="${HOST_EXTRA_DEFS:-} -DSQLITE_HC_REGION_SIZE=${REGION}UL"

# SPEEDTEST1_REGION_ARENA=1 takes SQLite's memsys5 arena out of .bss and into a THIRD shared region.
# Off by default, so every recorded result keeps the geometry it was measured with.
#
# WHY IT EXISTS. A .bss array is carved out of dom_data, and dom_data is ONE __get_free_pages
# allocation capped at 4 MiB on this kernel. That ceiling is what blocks `json`, whose arena need is
# ~6 MiB, and the whole --size axis. A region is CMA-backed and has been demonstrated at 130 MiB.
#
# ONLY THE HOST IS TOLD THE SIZE. The domain reads it off the grant's own bounds, so there is no
# second place for the two halves to disagree -- which is the failure mode SQLITE_HC_REGION_SIZE
# needs a runtime gate to catch.
if [[ "${SPEEDTEST1_REGION_ARENA:-0}" == "1" ]]; then
  ARENA=${SPEEDTEST1_ARENA_SIZE:-$SPEEDTEST1_GEOM_HEAP}
  DOMAIN_EXTRA_DEFS="$DOMAIN_EXTRA_DEFS -DCAPSTONE_SPEEDTEST1_REGION_ARENA=1"
  HOST_EXTRA_DEFS="$HOST_EXTRA_DEFS -DSPEEDTEST1_REGION_ARENA=1 -DSPEEDTEST1_ARENA_SIZE=${ARENA}UL"
  echo "== arena from a REGION: $ARENA bytes, shared third; .bss array not built"
fi
export DOMAIN_EXTRA_DEFS HOST_EXTRA_DEFS
export DOMAIN_SRC="$SCRIPT_DIR/speedtest1_measure.c"

# SPEEDTEST1_SUBLET=1: build the domain on the Sublet port of memsys5 and the lookaside pool, and
# lend it the pool and tables as REVOCABLE regions. This is the sixth matrix cell -- the same
# allocators with the discipline applied -- so lookaside stays on and only the discipline changes.
#
# THE REGION OPTIONS GO AS SEPARATE ARGV ELEMENTS, BEFORE the benchmark's quoted command line.
# Everything inside those quotes is handed to speedtest1 itself, so `--arena` written there is a
# speedtest1 flag, not a region request: the host lifts these pairs out of argv before the join,
# and it can only do that if they arrive as their own arguments.
#
# --arena, NOT SPEEDTEST1_REGION_ARENA: --arena shares REV_BORROWED, the linear borrow under a
# handle the monitor keeps, which is the revocable relationship the discipline is about.
# REV_SHARED is delinearised by the monitor and cannot carry it. The two claim the same slot and
# the host refuses both, so this is enforced rather than advised.
SUBLET_HOST_ARGS=""
if [[ "${SPEEDTEST1_SUBLET:-0}" == "1" ]]; then
  export SQLITE_SUBLET_PATCH=${SQLITE_SUBLET_PATCH:-$SCRIPT_DIR/sublet/sublet-3530300.patch}
  DOMAIN_EXTRA_DEFS="$DOMAIN_EXTRA_DEFS -DSPEEDTEST1_SUBLET=1"
  SUBLET_POOL=${SPEEDTEST1_POOL:-1441792}
  _atoms=$(( SUBLET_POOL / 65 ))
  SUBLET_ARENA=$(( _atoms * 64 ))
  SUBLET_TABLES=$(( _atoms * 57 + _atoms * 16 + 131072 ))
  SUBLET_HOST_ARGS=" --arena $SUBLET_ARENA --tables $SUBLET_TABLES"
  echo "== Sublet: pool $SUBLET_ARENA bytes (arena, REV_BORROWED), tables $SUBLET_TABLES bytes"
fi


bash "$SCRIPT_DIR/build-sqlite-silicon.sh"
bash "$SCRIPT_DIR/build-sqlite-host.sh"

SHARE=${SHARE_DIR:-$CAPSTONE_TMP_ROOT/sqlite-speedtest1-share}
rm -rf "$SHARE"; mkdir -p "$SHARE"
DOM="$OUT_DIR/sqlite_silicon.dom"
[[ -f "$DOM" ]] || { echo "ERROR: $DOM does not exist -- the build produced nothing" >&2; exit 1; }
cp -f "$DOM" "$SHARE/"
cp -f "$OUT_DIR/sqlite_host.user" "$SHARE/"
echo "== domain $(sha256sum "$DOM" | cut -c1-16)  region=$REGION  heap=$SQLITE_HEAP_SIZE  args='$SPEEDTEST1_ARGS'"

# WITHOUT -icount, mcycle UNDER QEMU IS THE HOST MACHINE'S TIMESTAMP COUNTER, not anything the guest
# did. target/riscv/csr.c:750-763 reads cpu_get_host_ticks() unless icount is enabled, so the number
# the domain reports is how long the x86 box took. It varies run to run, it cannot be compared with
# a board reading, and dividing it by 25 MHz mixes two unrelated units. With -icount it becomes a
# deterministic INSTRUCTION count, which is a quantity the board can be predicted from (board cycles
# are roughly CPI x instructions, and this core's measured ratios span 1.13 to 6.44).
#
# On by default, because the failure mode of forgetting it is a plausible-looking number that means
# nothing. SPEEDTEST1_ICOUNT=off gives the old behaviour for anyone who wants host wall time.
# When the invocation asks for --verify, REQUIRE the hash line. Without this the run passes on the
# three markers alone and the hash -- which is the verdict for main and orm -- could be absent
# without anyone noticing.
VERIFY_MARKER=()
case "$SPEEDTEST1_ARGS" in
  *--verify*) VERIFY_MARKER=(--success-marker 'Verification Hash:') ;;
esac

# SPEEDTEST1_QEMU_EXTRA=cma reserves a CMA area in the guest. The module allocates REGIONS with
# dma_alloc_pages (capstone.c:234), which takes CMA; CONFIG_CMA_SIZE_MBYTES is 0 and no cma= appears
# on any command line, so with nothing reserved it falls back to the buddy allocator and inherits the
# same order-10 (4 MiB) cap the domain region has. This exists to test whether the recorded "region
# route caps around 4 MiB" is architectural or just an unreserved area. QEMU takes the LAST -append,
# so root= is repeated here rather than lost.
CMA_ARGS=()
if [[ "${SPEEDTEST1_QEMU_EXTRA:-}" == "cma" ]]; then
  CMA_ARGS=(--qemu-extra-arg=-append --qemu-extra-arg="root=/dev/vda ro cma=${SPEEDTEST1_CMA_MB:-64}M")
  echo "== reserving a ${SPEEDTEST1_CMA_MB:-64} MiB CMA area in the guest"
fi

ICOUNT_ARGS=()
if [[ "${SPEEDTEST1_ICOUNT:-on}" != "off" ]]; then
  ICOUNT_ARGS=(--qemu-extra-arg=-icount --qemu-extra-arg="${ICOUNT:-shift=0,sleep=off}")
fi

# SPEEDTEST1-CYCLES is asserted alongside the RAN marker on purpose: the marker alone would also be
# printed by a run whose report was truncated away, and a measurement with no number in it is not a
# measurement. TOTAL comes from speedtest1 itself and is what says the benchmark reached its end.
#
# NOTE THE TOTAL IS NOT AN INDEPENDENT CHECK ON THE CYCLE COUNT. Both come from the same mcycle
# read -- speedtest1's per-phase times go through the mcycle-backed VFS clock -- so their agreement
# says the clock is PLUMBED, and says nothing whatever about whether it is scaled to anything real.
python3 "$ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
  --share-dir "$SHARE" \
  --log-file "${SPEEDTEST1_LOG_FILE:-$OUT_DIR/sqlite-speedtest1.log}" \
  --timeout-multiplier 12 \
  "${ICOUNT_ARGS[@]}" \
  "${CMA_ARGS[@]}" \
  --guest-command \
    "cp /mnt/host/sqlite_host.user /tmp/h.user && chmod 0755 /tmp/h.user && /tmp/h.user /mnt/host/sqlite_silicon.dom --speedtest1$SUBLET_HOST_ARGS '$SPEEDTEST1_ARGS'" \
  --success-marker 'SPEEDTEST1-CYCLES' \
  --success-marker '__CAPSTONE_SPEEDTEST1_RAN__' \
  --success-marker 'TOTAL' \
  "${VERIFY_MARKER[@]}"

echo "__CAPSTONE_SPEEDTEST1_QEMU_RAN__"
