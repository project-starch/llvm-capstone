#!/usr/bin/env bash
# Temporal-safety overhead of the OFFICIAL binary-trees Lua benchmark on Capstone.
#
# Builds TWO byte-identical binary-trees domains that differ only in whether free()
# revokes, and runs each once under `-icount shift=0` (deterministic dynamic
# instruction count). The pcall of the benchmark is bracketed by rdcycle; newstate +
# base + load are outside the bracket, so the delta is purely the workload:
#
#   revoke   : full revoke-on-free (Capstone's temporal safety — a revoke per free)
#   norevoke : identical allocator, revoke suppressed (rof_no_revoke=1)
#   revoke - norevoke = the O(1) revoke cost amortised over the whole benchmark
#
# This is the binary-trees counterpart to run-tree-cost-probe.sh (the BST probe that
# measured +10 instr/op), and the Capstone counterpart to the CHERI baseline in
# xlang/lua-cdp/cheri/bench/. Functional-model proxy, not cycle-accurate timing.
#
#   ./measure-bintrees-cost.sh            # N=6 (matches the CHERI reproduce script)
#   N=6 ./measure-bintrees-cost.sh
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

N=${N:-6}
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/lua-bt-cost}
SMOKE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
ICOUNT=${ICOUNT:-shift=0,sleep=off}
mkdir -p "$SHARE"

echo "== build revoke + norevoke binary-trees domains (N=$N) =="
OUT_DIR="$SHARE" OUT_DOM="$SHARE/lua_bt_revoke.dom" \
  DOMAIN_EXTRA_DEFS="-DLUA_BINTREES -DLUA_BT_N=$N -DLUA_DBG_STAGE" \
  bash "$HERE/build-lua-domain.sh" >"$SHARE/build_revoke.log" 2>&1 \
  || { echo "revoke build FAILED"; tail -5 "$SHARE/build_revoke.log"; exit 1; }
OUT_DIR="$SHARE" OUT_DOM="$SHARE/lua_bt_norevoke.dom" \
  DOMAIN_EXTRA_DEFS="-DLUA_BINTREES -DLUA_BT_NO_REVOKE -DLUA_BT_N=$N -DLUA_DBG_STAGE" \
  bash "$HERE/build-lua-domain.sh" >"$SHARE/build_norevoke.log" 2>&1 \
  || { echo "norevoke build FAILED"; tail -5 "$SHARE/build_norevoke.log"; exit 1; }

run_mode() { # $1 = revoke|norevoke
  local m="$1" log="$SHARE/bt_$1.log"
  for attempt in 1 2 3; do
    timeout 1200 python3 "$SMOKE" \
      --share-dir "$SHARE" --log-file "$log" --timeout-multiplier 16 \
      --qemu-extra-arg=-icount --qemu-extra-arg="$ICOUNT" \
      --guest-command "cp /mnt/host/lua_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/lua_bt_$m.dom" \
      --success-marker '__never__' >/dev/null 2>&1
    grep -qaE 'BT-OK .*icount=' "$log" && return 0
    echo "  $m attempt $attempt: no BT-OK marker, retrying" >&2
  done
  return 1
}

for m in norevoke revoke; do   # norevoke first: it never revokes, so it can't wedge on a revoke bug
  echo "== boot mode: $m =="
  run_mode "$m" || { echo "$m run FAILED; see $SHARE/bt_$m.log" >&2; exit 1; }
  grep -aoE 'BT-OK[^Z]*' "$SHARE/bt_$m.log" | tail -1
done

echo
echo "=== binary-trees temporal-safety overhead (N=$N, functional-model proxy) ==="
python3 - "$SHARE" <<'PY'
import re, sys, os
share = sys.argv[1]
def readlog(mode):
    return open(os.path.join(share, f"bt_{mode}.log"), encoding="utf-8", errors="replace").read()
def perf(log):
    m = re.search(r"BT-OK .*?check=(-?\d+) icount=(\d+)", log)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)
def mem(log):
    m = re.search(r"BT-MEM carved_bytes=(\d+) peak_live_objs=(\d+) end_live_objs=(\d+) end_live_bytes=(\d+)", log)
    n = re.search(r"REV-NODES alloced_n=(\d+)", log)
    return ((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))) if m else None,
            int(n.group(1)) if n else None)
lr, ln = readlog("revoke"), readlog("norevoke")
ck_r, ic_r = perf(lr); ck_n, ic_n = perf(ln)
if None in (ic_n, ic_r):
    print("  MISSING icount (see logs)"); sys.exit(1)
print("  -- PERFORMANCE (dynamic instruction count under -icount shift=0) --")
print(f"  check (both must match, correctness) : norevoke={ck_n}  revoke={ck_r}  {'OK' if ck_n==ck_r else 'MISMATCH!'}")
print(f"  norevoke (alloc-side only) icount    : {ic_n:,}")
print(f"  revoke   (full temporal)   icount    : {ic_r:,}")
print(f"  revoke - norevoke (revoke cost)      : +{ic_r-ic_n:,} instr  ({ic_r/ic_n:.4f}x)")
memr, nodes = mem(lr)
if memr:
    carved, peak, elo, elb = memr
    print("\n  -- MEMORY (revoke build) --")
    print(f"  heap: peak_live_objs={peak:,}  end_live={elo:,} objs / {elb:,} B  carved_total={carved:,} B (rof never reclaims)")
    if nodes is not None:
        print(f"  revocation-node metadata high-water  : {nodes:,} nodes (~{nodes*20//1024} KB @ ~20 B/node)")
        print(f"  (nodes/alloc ~{nodes/max(peak,1):.1f}x working set -> leaked; a reclaiming allocator would bound to ~2x peak)")
PY
echo "logs: $SHARE/bt_{revoke,norevoke}.log"
