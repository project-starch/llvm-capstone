#!/usr/bin/env bash
# The unit test of docs/plans/2026-11-15-php-port-layout-fix.md: probe-layout.c against the current
# zend-cap-mode.c and, as the negative control, against the file at 5060011844e, in one boot.
#
#   test-probe-layout.sh SDK KIT OLD WORK
#
# SDK   a virtual SDK directory holding capstone-cc (for example a build root's sdk/)
# KIT   a trap-route kit (kit-run-3 on p13)
# OLD   a directory holding zend-cap-mode.c and zend-cap-mode.h at 5060011844e
#       (git -c submodule.recurse=false show 5060011844e:capstone/ports/php/zend-alloc/sublet/zend-cap-mode.c)
# WORK  a new directory for the images, the boot and verdict.txt
#
# Run from the repository root. Exits 0 when every stage reads as the plan predicts: the new build
# passes geometry, realloc and big and faults with cause 25 on the stale accesses and 28 past the
# bound; the old build fails geometry and realloc and still faults with cause 25.
set -euo pipefail
[[ $# == 4 ]] || { echo "usage: test-probe-layout.sh SDK KIT OLD WORK" >&2; exit 2; }
SDK=$(realpath "$1") KIT=$(realpath "$2") OLD=$(realpath "$3") WORK=$(realpath -m "$4")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$HERE/../../../../.." && pwd)
[[ ! -e $WORK ]] || { echo "use a new WORK: $WORK" >&2; exit 2; }
mkdir -p "$WORK/old-src"
cp "$OLD/zend-cap-mode.c" "$OLD/zend-cap-mode.h" "$WORK/old-src/"

WRAP=(-Wl,--wrap=malloc -Wl,--wrap=realloc -Wl,--wrap=free)
INC=(-I"$R/capstone" -I"$R/capstone/runtime/include")
"$SDK/capstone-cc" -O1 "${INC[@]}" -I"$HERE" "$HERE/probe-layout.c" "$HERE/zend-cap-mode.c" \
  "${WRAP[@]}" -o "$WORK/new.dom"
"$SDK/capstone-cc" -O1 "${INC[@]}" -I"$WORK/old-src" "$HERE/probe-layout.c" "$WORK/old-src/zend-cap-mode.c" \
  "${WRAP[@]}" -o "$WORK/old.dom"

STAGES="geometry realloc stale-realloc stale-free bound big stale-big"
ARGS=()
for build in new old; do
  ARGS+=(--file "$build.dom=$WORK/$build.dom")
  for s in $STAGES; do
    ARGS+=(--step "$build-$s=cd /mnt/vm; ./capstone-vexec ./$build.dom $s")
  done
done
python3 "$R/capstone/runtime/virtual/trap_platform.py" run --kit "$KIT" --work "$WORK/run" \
  --step-timeout 600 "${ARGS[@]}"

set +e
python3 - "$WORK/run/results.json" > "$WORK/verdict.txt" <<'EOF'
import json, re, sys
results = {r['name']: r for r in json.load(open(sys.argv[1]))}

def cause(r):
    m = re.search(r'cause=(\d+)', (r.get('output') or '') + json.dumps(r.get('fault')))
    return int(m.group(1)) if m else None

def passes(r, word):
    return r['kind'] == 'exit' and r['value'] == 0 and word in r['output']

checks = [
    ('new-geometry', lambda r: passes(r, 'GEOMETRY_PASS'), 'GEOMETRY_PASS, exit 0'),
    ('old-geometry', lambda r: 'GEOMETRY_FAIL' in r['output'], 'GEOMETRY_FAIL (negative control)'),
    ('new-realloc', lambda r: passes(r, 'REALLOC_PASS'), 'REALLOC_PASS, exit 0'),
    ('old-realloc', lambda r: 'REALLOC_FAIL' in r['output'], 'REALLOC_FAIL (negative control)'),
    ('new-stale-realloc', lambda r: 'STALE_REALLOC_READY inplace=1' in r['output'] and cause(r) == 25,
     'in place, then cause 25'),
    ('old-stale-realloc', lambda r: 'STALE_REALLOC_READY' in r['output'] and cause(r) == 25, 'cause 25'),
    ('new-stale-free', lambda r: 'STALE_FREE_READY' in r['output'] and cause(r) == 25, 'cause 25'),
    ('old-stale-free', lambda r: 'STALE_FREE_READY' in r['output'] and cause(r) == 25, 'cause 25'),
    ('new-bound', lambda r: 'BOUND_READY' in r['output'] and cause(r) == 28, 'cause 28'),
    ('new-big', lambda r: passes(r, 'BIG_PASS'), 'BIG_PASS, exit 0'),
    ('new-stale-big', lambda r: 'STALE_BIG_READY' in r['output'] and cause(r) == 25, 'cause 25'),
]
ok = True
for name, check, want in checks:
    r = results[name]
    good = check(r)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name}: want {want}; got {r['kind']} {r['value']} cause={cause(r)}")
for name in sorted(results):
    for line in results[name]['output'].splitlines():
        if re.match(r'(GEOMETRY|REALLOC|BIG|STALE|BOUND)', line):
            print(f"  {name}: {line}")
print('UNIT TEST', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
EOF
rc=$?
cat "$WORK/verdict.txt"
exit $rc
