#!/usr/bin/env bash
# Reproduce the Lua-CDP CHERI-vs-Capstone security comparison and CHECK it.
#
#   ./reproduce.sh            # both columns, rebuild + re-measure + verify (~1 h)
#   ./reproduce.sh capstone   # Capstone column only (13 rows x {revoke,control})
#   ./reproduce.sh cheri      # CHERI column only (13 rows x {spatial,async,eager})
#
# This is a CHECK, not just a runner: it re-runs from source and EXITS NON-ZERO on
# any disagreement with capstone/expected-results.tsv or cheri/expected-results.tsv.
# A column nobody can re-measure is a claim, not a result.
#
# Both columns rebuild every shim from xlang/lua-cdp/shims/ — the SAME source — so
# this also validates that one shim set drives both platforms.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET=${1:-both}
rc_total=0

check_capstone() {
  echo "== Capstone column: rebuild + re-measure (neutral allocator) =="
  CAPSTONE_REPO_ROOT=${CAPSTONE_REPO_ROOT:-$(cd "$HERE/../.." && pwd)} \
    bash "$HERE/capstone/run-lua-capstone.sh" || { echo "capstone run failed" >&2; return 1; }
  local results="${CAPSTONE_TMP_ROOT:-/tmp/capstone}/lua-cdp-capstone/results.tsv"
  echo "== verify against capstone/expected-results.tsv =="
  python3 - "$HERE/capstone/expected-results.tsv" "$results" <<'PY'
import sys
exp_path, act_path = sys.argv[1], sys.argv[2]
exp = {}
for ln in open(exp_path):
    if ln.startswith('#') or ln.startswith('key\t'): continue
    k, row, rev, ctl, mech = ln.rstrip('\n').split('\t')
    exp[row] = (rev, ctl)
act = {}
for ln in open(act_path):
    f = ln.rstrip('\n').split('\t')
    if len(f) < 3 or f[0] == 'row': continue
    act.setdefault(f[0], {})[f[1]] = f[2]     # row -> {variant: outcome}
bad = []
for row, (rev, ctl) in exp.items():
    a = act.get(row, {})
    if a.get('revoke') != rev or a.get('control') != ctl:
        bad.append(f"  {row}: expected revoke={rev}/control={ctl}, "
                   f"got revoke={a.get('revoke')}/control={a.get('control')}")
if bad:
    print(f"CAPSTONE FAILED ({len(bad)}/{len(exp)} rows differ)"); print('\n'.join(bad)); sys.exit(1)
print(f"CAPSTONE REPRODUCED  {len(exp)}/{len(exp)} rows identical to expected")
PY
}

check_cheri() {
  echo "== CHERI column: rebuild + re-measure (spatial/async/eager) =="
  CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri} \
    bash "$HERE/cheri/run-lua-cheri.sh" || echo "  (cheri-run may report a post-run TIMEOUT; the serial log is what we verify)"
  local serial="${CHERI_ROOT:-$HOME/cheri}/xlang-run/serial.log"
  echo "== verify against cheri/expected-results.tsv =="
  python3 - "$HERE/cheri/expected-results.tsv" "$serial" "$HERE/cheri/rows.tsv" <<'PY'
import sys, re
exp_path, serial_path, rows_path = sys.argv[1], sys.argv[2], sys.argv[3]
# key -> shim stem (numeric key <-> row name)
name = {}
for ln in open(rows_path):
    if ln.startswith('#'): continue
    f = ln.rstrip('\n').split('\t')
    if len(f) >= 2 and f[0].strip().isdigit():
        name[f[0]] = f[1].split('/')[-1][:-2] if f[1].endswith('.c') else f[1]
exp = {}
for ln in open(exp_path):
    if ln.startswith('#') or ln.startswith('key\t'): continue
    f = ln.rstrip('\n').split('\t')          # key row spatial async eager cdp
    exp[f[0]] = {'spatial': f[2], 'async': f[3], 'eager': f[4]}
def cls(rc):
    return {0:'exit0', 162:'SIGPROT', 134:'SIGABRT'}.get(int(rc), f'rc{rc}')
act = {}   # key -> {cfg: outcome}   (serial cfg 'temporal' maps to 'async')
text = open(serial_path, errors='replace').read()
for m in re.finditer(r"ROW (\d+) cfg=(\w+) rc=(\d+)", text):
    cfg = 'async' if m.group(2) == 'temporal' else m.group(2)
    act.setdefault(m.group(1), {})[cfg] = cls(m.group(3))
bad = []
for key, e in exp.items():
    a = act.get(key, {})
    for cfg in ('spatial', 'async', 'eager'):
        if a.get(cfg) != e[cfg]:
            bad.append(f"  key {key} ({name.get(key,'?')}) {cfg}: expected {e[cfg]}, got {a.get(cfg)}")
if bad:
    print(f"CHERI FAILED ({len(bad)} cells differ)"); print('\n'.join(bad)); sys.exit(1)
print(f"CHERI REPRODUCED  {len(exp)}/{len(exp)} rows x 3 configs identical to expected")
PY
}

case "$TARGET" in
  capstone) check_capstone || rc_total=1 ;;
  cheri)    check_cheri    || rc_total=1 ;;
  both)     check_capstone || rc_total=1; echo; check_cheri || rc_total=1 ;;
  *) echo "usage: $0 [capstone|cheri|both]" >&2; exit 2 ;;
esac

echo
[ "$rc_total" -eq 0 ] && echo "== reproduce.sh: PASS ==" || echo "== reproduce.sh: FAIL =="
exit "$rc_total"
