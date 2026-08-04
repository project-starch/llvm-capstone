#!/usr/bin/env bash
# CHECK tarantool #7657 by DIFFERENTIAL: the vulnerable image must SIGSEGV in the
# merge_source gen path (tarantool crash handler fires, docker exit 139); the
# fixed image must complete and print 7000. PASS only if both hold. This is what
# makes it a verified reproduction, not a lucky crash.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"; mkdir -p "$W"
VULN_IMG=tarantool/tarantool:2.8.3
FIX_IMG=tarantool/tarantool:2.11
TRIG="$HERE/trigger.lua"

command -v docker >/dev/null || { echo "docker required" >&2; exit 2; }
docker image inspect "$VULN_IMG" >/dev/null 2>&1 || { echo "run ./build.sh first" >&2; exit 2; }

run_trigger() { # $1 = image -> captures stdout+stderr and the docker exit code
  timeout 240 docker run --rm --entrypoint tarantool \
    -v "$TRIG:/trigger.lua:ro" "$1" /trigger.lua 2>&1
}

echo "== vulnerable ($VULN_IMG) =="
V=$(run_trigger "$VULN_IMG"); VC=$?; echo "$V" | tail -6; echo "  (exit $VC)"
echo "== fixed ($FIX_IMG) =="
F=$(run_trigger "$FIX_IMG"); FC=$?; echo "$F" | tail -3; echo "  (exit $FC)"

# Vulnerable: SIGSEGV. docker relays 128+SIGSEGV(11) = 139, and tarantool's crash
# handler prints its fault report ("Segmentation fault ... Please file a bug").
vuln_uaf=0
{ [ "$VC" = 139 ] || echo "$V" | grep -qaE 'Segmentation fault|crash_signal_cb|Please file a bug'; } \
  && echo "$V" | grep -qaE 'lj_BC_FUNCC|SEGV_MAPERR|addr: 0' && vuln_uaf=1
# Fixed: clean completion, prints the expected 7000, no crash report.
fix_clean=0
{ [ "$FC" = 0 ] && echo "$F" | grep -qa '7000' && ! echo "$F" | grep -qaE 'Segmentation fault|Please file a bug'; } \
  && fix_clean=1

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: tarantool #7657 reproduced (SIGSEGV on 2.8.3, clean 7000 on 2.11)"; exit 0
fi
echo "FAIL: differential not satisfied" >&2; exit 1
