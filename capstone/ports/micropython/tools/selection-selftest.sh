#!/usr/bin/env bash
# Negative test for the SELECTION rule in gen-test-table.py. Needs no board, no QEMU and no
# build: it generates a table from synthetic tests and requires the right ones to be refused.
#
# The rule this guards replaced a substring match on "import ", which excluded 83 tests of which
# 73 imported nothing but modules the port had built in. A selection that has never refused
# anything is not a narrow selection, it is an unmeasured one, so every case below states both
# halves: what must be kept and what must be dropped, and with which reason.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEN=$HERE/gen-test-table.py
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
T=$WORK/tests; mkdir -p "$T"

cat > "$T/plain.py"        <<'PY'
print(1 + 1)
PY
cat > "$T/have_sys.py"     <<'PY'
import sys
print(hasattr(sys, "argv"))
PY
cat > "$T/absent_mod.py"   <<'PY'
import nosuchmodule
print(nosuchmodule)
PY
cat > "$T/from_absent.py"  <<'PY'
from nosuchpackage.inner import thing
print(thing)
PY
cat > "$T/excluded_mod.py" <<'PY'
import uctypes
print(uctypes)
PY
# types exists on the host and not on the target, which is the whole shape: the oracle runs the
# else-branch, the target never does, and a selection that reads the import literally drops a test
# for a module the target is never asked for.
cat > "$T/cpython_only.py" <<'PY'
import sys
if sys.implementation.name == 'micropython':
    helper = lambda f: f
else:
    import types
    helper = types.FunctionType
print(callable(helper))
PY
cat > "$T/guarded_try.py"  <<'PY'
try:
    import nosuchmodule
except ImportError:
    print("SKIP")
    raise SystemExit
print(nosuchmodule)
PY
cat > "$T/relative.py"     <<'PY'
from . import sibling
print(sibling)
PY
cat > "$T/uses_float.py"   <<'PY'
print(1.0 + 1.0)
PY
cat > "$T/uses_thread.py"  <<'PY'
import _thread
print(_thread)
PY
{ echo "# padding to exceed a small size limit"; for i in $(seq 40); do
    echo "print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')"; done; } > "$T/oversize.py"
# Syntax this host python cannot parse, so the regex fallback is the only thing that sees the
# import. Without the fallback the file would reach the oracle and be dropped for the wrong reason.
printf 'import nosuchmodule\nawait_ = 1 if else 2\n' > "$T/unparsable.py"

run() {  # run the generator; $* = extra args
  python3 "$GEN" "$T" "$WORK/out.h" --have-module sys --exclude-module uctypes "$@" >/dev/null 2>&1
}
reason() {  # the recorded skip reason for $1, or KEPT
  if grep -qE "^    /\* $1 \*/" "$WORK/out.h"; then echo KEPT
  else sed -n "s@^ \*   $1 *@@p" "$WORK/out.h" | head -1; fi
}

fail=0
expect() {  # $1 = test file, $2 = expected reason or KEPT
  local got; got=$(reason "$1")
  if [[ $got == "$2" ]]; then printf "  ok    %-20s %s\n" "$1" "$got"
  else printf "  FAIL  %-20s got '%s', wanted '%s'\n" "$1" "$got" "$2"; fail=1; fi
}

echo "== no size limit, no float"
run --max-bytes 0
expect plain.py        KEPT
expect have_sys.py     KEPT
expect cpython_only.py KEPT
expect oversize.py     KEPT
expect absent_mod.py   "needs module nosuchmodule"
expect from_absent.py  "needs module nosuchpackage"
expect excluded_mod.py "needs module uctypes (excluded)"
expect guarded_try.py  "needs module nosuchmodule"
expect unparsable.py   "needs module nosuchmodule"
expect relative.py     "host python exits non-zero"
expect uses_float.py   "needs float"
expect uses_thread.py  "needs module _thread"

echo "== with a size limit"
run --max-bytes 200
expect oversize.py     "larger than 200 B"
expect plain.py        KEPT

echo "== with float"
run --max-bytes 0 --have-float
expect uses_float.py   KEPT

if [[ $fail == 0 ]]; then echo "selection-selftest: every case behaved"; else echo "selection-selftest: FAILED"; fi
exit $fail
