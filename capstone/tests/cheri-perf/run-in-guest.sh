#!/bin/sh
# Runs INSIDE CheriBSD purecap (as root) under CHERI-QEMU. Measures the
# temporal-safety overhead of a malloc/touch/free loop under the three CHERI
# revocation policies, printing one RCPERF line per (config, trial) that the host
# driver parses. This is the CHERI arm of the CHERI-vs-Capstone perf comparison
# (plans/perf-cheri-vs-capstone-qemu.md).
#
#   spatial  : revocation OFF                      -> CHERI spatial safety only (baseline)
#   temporal : revocation ON, async quarantine     -> realistic CHERI temporal deployment
#   eager    : revocation ON, revoke on every free -> aggressive, matches our security
DIR=/root/cheri-perf
ITERS="${RC_ITERS:-200000}"
BLOCK="${RC_BLOCK:-64}"
TRIALS="${RC_TRIALS:-3}"
cd "$DIR" || exit 2

# --- pick a user-readable instruction counter (rdinstret preferred) -----------
BIN=./rc_instret
if ! ./rc_instret probe 2048 "$BLOCK" >/dev/null 2>&1; then
  echo "RCPERF-NOTE rdinstret unreadable in userspace, falling back to rdcycle"
  BIN=./rc_cycle
  if ! ./rc_cycle probe 2048 "$BLOCK" >/dev/null 2>&1; then
    echo "RCPERF-FATAL neither rdinstret nor rdcycle is user-readable"
    echo "===CHERI-PERF-END==="
    exit 3
  fi
fi
echo "RCPERF-COUNTER $BIN"

set_cfg() { # $1 = spatial|temporal|eager
  case "$1" in
    spatial)  REV=0; EVERY=0 ;;
    temporal) REV=1; EVERY=0 ;;
    eager)    REV=1; EVERY=1 ;;
    *) echo "unknown cfg $1"; return 2 ;;
  esac
  sysctl security.cheri.runtime_revocation_default=$REV >/dev/null 2>&1
  sysctl security.cheri.runtime_revocation_every_free_default=$EVERY >/dev/null 2>&1
  echo "KNOB cfg=$1 rev=$(sysctl -n security.cheri.runtime_revocation_default 2>/dev/null) every=$(sysctl -n security.cheri.runtime_revocation_every_free_default 2>/dev/null)"
  [ -x ./cheri_status ] && echo "STATUS cfg=$1 $(./cheri_status 2>&1 | tr '\n' ' ')"
}

echo "===CHERI-PERF-BEGIN iters=$ITERS block=$BLOCK trials=$TRIALS==="
if [ "${RC_SKIP_MICRO:-0}" = 1 ]; then
  echo "RCPERF-NOTE microbench skipped (RC_SKIP_MICRO=1)"
else
for cfg in spatial temporal eager; do
  set_cfg "$cfg"
  t=1
  while [ "$t" -le "$TRIALS" ]; do
    "$BIN" "$cfg" "$ITERS" "$BLOCK"
    t=$((t + 1))
  done
done
fi

# Real-workload arm: the binary-search-tree build/lookup/destroy workload.
# Each free is a revocation point; under eager that is a full sweep, so eager
# gets ONE trial (still ~20k sweeps) while spatial/async get the normal count.
TREEBIN=./rc_tree
[ "$BIN" = ./rc_cycle ] && TREEBIN=./rc_tree_cycle
echo "RCTREE-BIN $TREEBIN"
for cfg in spatial temporal eager; do
  set_cfg "$cfg"
  tt="$TRIALS"; [ "$cfg" = eager ] && tt=1
  t=1
  while [ "$t" -le "$tt" ]; do
    "$TREEBIN" "$cfg"
    t=$((t + 1))
  done
done
echo "===CHERI-PERF-END==="
