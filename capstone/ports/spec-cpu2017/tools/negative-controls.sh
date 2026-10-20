#!/bin/bash
# The two checks that show the evidence chain can fail. Both must come out as expected.
# usage: negative-controls.sh KIT SPEC_ROOT TOOLCHAIN PLATFORM BUILD_RECORD OUT
#   1. tampered output: a copy of the kit whose wrapper appends one line to every guest
#      invocation's standard output; runcpu's validation must reject 997.specrand_fr (test) and
#      tools/verdict.py must say FAIL.
#   2. foreign executable: the wrapper must refuse a program that is neither a Capstone image nor
#      one of SPEC's tools (here the host's /bin/true), exit 125 and record "refused".
# Writes OUT/negative-controls.txt; exits non-zero unless both controls behave.
set -euo pipefail
KIT=$(cd "$1" && pwd) SPEC_ROOT=$(cd "$2" && pwd) TC=$3 PL=$4 BREC=$5 OUT=$6
[ -e "$OUT" ] && { echo "negative-controls: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
R=$OUT/negative-controls.txt; : > "$R"
ok=1
# 1. Tampered output.
cp -r "$KIT/tools" "$KIT/config" "$KIT/manifest.lock.json" "$OUT/"
mkdir -p "$OUT/tampered-kit"; mv "$OUT/tools" "$OUT/config" "$OUT/manifest.lock.json" "$OUT/tampered-kit/"
python3 - "$OUT/tampered-kit/tools/capstone-run.sh" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = 'record guest "$rc"'
assert s.count(a) == 1
s = s.replace(a, 'echo "negative control: one extra line"\nrecord guest "$rc"')
open(p, 'w').write(s)
PY
bash "$KIT/tools/run.sh" "$OUT/tampered-kit" "$SPEC_ROOT" "$TC" "$PL" "$BREC" test 997.specrand_fr "$OUT/tampered" > "$OUT/tampered.out" 2>&1 || true
v=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['verdict'],'|',';'.join(d['reasons']))" "$OUT/tampered/result.json" 2>&1)
case "$v" in
  FAIL*"SPEC reports no success"*) echo "tampered output: rejected as expected ($v)" >> "$R" ;;
  *) echo "tampered output: NOT rejected ($v)" >> "$R"; ok=0 ;;
esac
# 2. Foreign executable.
mkdir -p "$OUT/foreign"
rc=0
( cd "$OUT/foreign" && env SPEC="$SPEC_ROOT" CAPSTONE_SPEC_RECORD="$OUT/foreign" CAPSTONE_SPEC_KIT="$KIT" \
    bash "$KIT/tools/capstone-run.sh" /bin/true ) > "$OUT/foreign.out" 2>&1 || rc=$?
route=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['route'])" "$OUT/foreign/invocations/0000.json" 2>&1)
if [ "$rc" = 125 ] && [ "$route" = refused ]; then
  echo "foreign executable: refused as expected (exit $rc, route $route)" >> "$R"
else
  echo "foreign executable: NOT refused (exit $rc, route $route)" >> "$R"; ok=0
fi
cat "$R"
[ $ok = 1 ]
