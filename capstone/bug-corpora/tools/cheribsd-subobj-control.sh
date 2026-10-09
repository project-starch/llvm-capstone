# Sourced by the CheriBSD runners. Only when CHERI_EXTRA_CFLAGS is set -- i.e. on another arm built
# from the same runner -- it adds a field-crossing control to cases.json and, after the boot,
# refuses the suite unless that control died by SIGPROT. Unset, both functions do nothing, so the
# runner's own arm is unchanged.
subobj_control_add() {   # subobj_control_add <out> <sdk> <cflags...>
  local out=$1 sdk=$2; shift 2
  [ -n "${CHERI_EXTRA_CFLAGS:-}" ] || return 0
  "$sdk/bin/clang" "$@" ${CHERI_EXTRA_CFLAGS} "$(dirname "${BASH_SOURCE[0]}")/subobj-control.c" \
    -o "$out/bin/subobj-control" || { echo "CONTROL-FAILED subobj-control build" >&2; return 75; }
  python3 - "$out/cases.json" "$out/bin/subobj-control" <<'PY'
import json, sys
p, prog = sys.argv[1], sys.argv[2]
cases = json.load(open(p))
cases.insert(0, dict(name='subobj-control', program=prog, args=[], timeout=120,
                     expect_regex=r'.*', exit=162))
open(p, 'w').write(json.dumps(cases, indent=2) + '\n')
PY
}
subobj_control_check() {   # subobj_control_check <out>
  local out=$1
  [ -n "${CHERI_EXTRA_CFLAGS:-}" ] || return 0
  python3 - "$out/run/summary.json" <<'PY' || { echo "CONTROL-FAILED subobj-control did not die by SIGPROT: the flags are not shown to narrow a field in this boot" >&2; return 75; }
import json, sys
rows = json.load(open(sys.argv[1]))['results']
hit = [r for r in rows if isinstance(r, dict) and r.get('name') == 'subobj-control']
sys.exit(0 if hit and hit[0].get('exit') == 162 else 1)
PY
}
