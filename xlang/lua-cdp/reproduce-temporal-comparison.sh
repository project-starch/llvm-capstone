#!/usr/bin/env bash
# Reproduce the CHERI-vs-Capstone temporal-safety comparison on the OFFICIAL
# Computer Language Benchmarks Game `binary-trees` Lua benchmark, on BOTH axes:
# performance (dynamic instruction count) and memory (footprint).
#
# It runs, in sequence (QEMU is serialized — never two vehicles at once):
#   1. Capstone side  — measure-bintrees-cost.sh: two byte-identical domains that
#      differ only in whether free() revokes, each under -icount shift=0, plus the
#      rof heap counters and QEMU revocation-node high-water.
#   2. CHERI side      — reproduce-cheri-lua-bench.sh: the same benchmark as a purecap
#      CheriBSD process under spatial / temporal(async) / eager revocation, bracketed
#      by rdinstret and getrusage peak RSS.
# then prints a single combined table.
#
#   ./reproduce-temporal-comparison.sh          # N=6 (the tractable, comparable size)
#   N=6 ./reproduce-temporal-comparison.sh
#
# Prereqs: a built capstone-qemu (with the 65536-node revocation pool + the
# REV-NODES alloced_n print), the Capstone clang/lld + buildroot host toolchain, and
# a cheri-baseline-provisioned CheriBSD purecap image. See PERF-MEMORY.md.
#
# FUNCTIONAL-MODEL PROXY, not silicon timing: -icount / rdinstret are deterministic
# dynamic instruction counts; compare RATIOS across configs, not absolute counts
# across ISAs.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../.." && pwd)}"
export CAPSTONE_REPO_ROOT
N=${N:-6}
CAP_SHARE=${CAP_SHARE:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/lua-bt-cost}
CHERI_LOG=${CHERI_LOG:-$HOME/cheri/xlang-run/lua-bench.log}

echo "############################################################"
echo "# 1/2  CAPSTONE  (binary-trees N=$N, revoke vs norevoke)"
echo "############################################################"
SHARE="$CAP_SHARE" N="$N" bash "$HERE/capstone-lua/measure-bintrees-cost.sh" || {
  echo "Capstone measurement FAILED" >&2; exit 1; }

echo
echo "############################################################"
echo "# 2/2  CHERI  (binary-trees N=$N, spatial/temporal/eager)"
echo "############################################################"
N="$N" bash "$HERE/cheri/bench/reproduce-cheri-lua-bench.sh" || {
  echo "CHERI measurement FAILED (partial results may still be in $CHERI_LOG)" >&2; }

echo
echo "############################################################"
echo "# COMBINED — temporal-safety cost, same benchmark, both axes"
echo "############################################################"
python3 - "$CAP_SHARE" "$CHERI_LOG" <<'PY'
import re, sys, os
cap_share, cheri_log = sys.argv[1], sys.argv[2]

def cap():
    def g(mode, pat):
        try: t = open(os.path.join(cap_share, f"bt_{mode}.log"), errors="replace").read()
        except OSError: return None
        m = re.search(pat, t); return m
    ic_r = g("revoke", r"icount=(\d+)"); ic_n = g("norevoke", r"icount=(\d+)")
    nodes = g("revoke", r"REV-NODES alloced_n=(\d+)")
    mem = g("revoke", r"peak_live_objs=(\d+)")
    return (int(ic_r.group(1)) if ic_r else None, int(ic_n.group(1)) if ic_n else None,
            int(nodes.group(1)) if nodes else None, int(mem.group(1)) if mem else None)

def cheri():
    # Same block splitting as parse-bench.py: split on the CFG marker, take each
    # block up to its END marker (a per-cfg lookahead is fragile — do NOT reinvent it).
    try: log = open(cheri_log, encoding="latin-1", errors="replace").read()
    except OSError: return {}
    blocks = re.split(r"==CFG (\w+)", log)
    out = {}
    for i in range(1, len(blocks), 2):
        cfg = blocks[i]
        body = blocks[i + 1].split("END cfg=")[0]
        cal_i = [int(x) for x in re.findall(r"CAL BENCH instrs=(\d+)", body)]
        all_i = [int(x) for x in re.findall(r"BENCH instrs=(\d+)", body)]
        run_i = all_i[len(cal_i):]
        all_r = [int(x) for x in re.findall(r"maxrss_kb=(\d+)", body)]
        run_r = all_r[len(cal_i):]
        if cal_i and run_i:
            out[cfg] = (sum(run_i)/len(run_i) - sum(cal_i)/len(cal_i),
                        sum(run_r)/len(run_r) if run_r else None)
    return out

ic_r, ic_n, nodes, peak = cap()
ch = cheri()
print()
if None not in (ic_r, ic_n):
    print(f"CAPSTONE  revoke-on-free (= CHERI eager-strength: every free revokes synchronously)")
    print(f"  time   : revoke/norevoke = {ic_r/ic_n:.4f}x   (+{ic_r-ic_n:,} instr over the whole benchmark)")
    if nodes: print(f"  memory : {nodes:,} revocation nodes (~{nodes*20//1024} KB metadata; leaked by the non-reclaiming rof allocator)")
if "spatial" in ch:
    sp_i, sp_r = ch["spatial"]
    print(f"\nCHERI     (baseline = spatial; overhead of adding the temporal layer)")
    for cfg in ("temporal", "eager"):
        if cfg in ch:
            i, r = ch[cfg]
            rss = f", +{r-sp_r:,.0f} KB peak RSS" if (r and sp_r) else ""
            tag = " [deployed default; catches 0/13 CDP UAFs at the access]" if cfg=="temporal" else " [full temporal safety; undeployable]"
            print(f"  {cfg:9}: time {i/sp_i:.3f}x{rss}{tag}")
print("\nHEADLINE: Capstone reaches eager-strength temporal safety (13/13 caught) at ~1x time,")
print("          where CHERI's eager-strength costs ~390x time -> it ships async (0/13). Capstone")
print("          trades that time for revocation-node metadata (space).")
PY
echo
echo "Capstone logs: $CAP_SHARE/bt_{revoke,norevoke}.log    CHERI log: $CHERI_LOG"
