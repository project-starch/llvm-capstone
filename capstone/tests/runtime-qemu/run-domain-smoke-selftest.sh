#!/usr/bin/env bash
# Self-test of run-domain-smoke.py, the runner every QEMU domain test goes through.
#
#   bash run-domain-smoke-selftest.sh    exit 0 only if all four hold:
#     1. preflight  an image referencing an undefined weak symbol is refused before
#                   any boot, naming the symbol (C-56: its address is not NULL in a
#                   domain);
#     2. prompt     a boot whose guest commands print "# " -- a comment, a sed
#                   expression -- still passes;
#     3. long       in the same boot, one guest command of more than 1 KiB runs whole;
#     4. control    the same boot through the runner of 40eefa09420c (dev before this;
#                   pinned) FAILS: it takes the first "# " for the prompt.
#
# Needs CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld, llvm-nm), CAPSTONE_BUILDROOT_DIR and
# CAPSTONE_QEMU_BINARY, and a python with pexpect. Takes the QEMU lock per boot.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../capstone-test-env.sh" >/dev/null
REPO=$CAPSTONE_REPO_ROOT
OUT=${OUT:-$CAPSTONE_TMP_ROOT/domain-smoke-selftest}
PYTHON=${PYTHON:-python3}
rm -rf "$OUT"; mkdir -p "$OUT/share" "$OUT/weak"
verdict=0

# 1. preflight: a two-line domain with an undefined weak reference.
cat > "$OUT/weak/weak.c" <<'C'
extern int hook(int) __attribute__((weak));
int capstone_main(void) { return hook ? hook(1) : 7; }
C
"$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m -ffreestanding \
  -O1 -c "$OUT/weak/weak.c" -o "$OUT/weak/weak.o"
"$CAPSTONE_LD_LLD" -T "$REPO/capstone/my_first_domain/link.ld" -e 0 -o "$OUT/weak/weak.dom" "$OUT/weak/weak.o"
set +e
"$PYTHON" "$HERE/run-domain-smoke.py" --share-dir "$OUT/weak" --log-file "$OUT/weak/run.log" \
  --guest-command true > "$OUT/weak/out.txt" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]] && grep -q "undefined weak symbols" "$OUT/weak/out.txt" && grep -q "weak.dom: hook" "$OUT/weak/out.txt" \
   && [[ ! -s "$OUT/weak/run.log" ]]; then
  echo "  preflight: refuses weak.dom (hook) before booting"
else
  echo "  preflight: did NOT refuse the image before a boot (rc=$rc)"; sed 's/^/    /' "$OUT/weak/out.txt" | tail -5; verdict=1
fi

# 2 + 3. prompt and long command, one boot. The long command is 1500 bytes of
# no-ops between two markers; typed whole it would be cut at ~1 KiB.
pad=$(printf ': ; %.0s' $(seq 1 380))   # separate no-op commands, not one ':' with arguments
LONG="echo LONG-BEGIN; $pad echo LONG-END; echo SELFTEST-OK"
boot() { # runner log
  set +e
  capstone_with_qemu_lock "$PYTHON" "$1" --share-dir "$OUT/share" --log-file "$2" \
    --timeout-multiplier 4 \
    --guest-command "echo 'PROMPT-LIKE # a comment'; echo 's/# /#_/g'; echo PROMPT-DONE; echo SELFTEST-OK" \
    --guest-command "$LONG" \
    --success-marker SELFTEST-OK > "$2.out" 2>&1
  local r=$?
  set -e
  return $r
}
# The runner checks its success markers against EVERY guest command's output, so
# both commands print SELFTEST-OK; what each one must show is checked in the log.
if boot "$HERE/run-domain-smoke.py" "$OUT/test.log" &&
   grep -aq "^PROMPT-DONE" "$OUT/test.log" && grep -aq "^LONG-END" "$OUT/test.log"; then
  echo "  prompt + long (${#LONG}-byte command): PASS"
else
  echo "  prompt + long: FAIL"; tail -5 "$OUT/test.log.out" | sed 's/^/    /'; verdict=1
fi

# 4. control: the same boot through the old runner.
git -C "$REPO" show 40eefa09420c:capstone/tests/runtime-qemu/run-domain-smoke.py > "$OUT/old-runner.py"
grep -q 'qemu.expect(r"# "' "$OUT/old-runner.py" \
  || { echo "40eefa09420c's runner does not match \"# \"; it controls nothing" >&2; exit 2; }
if boot "$OUT/old-runner.py" "$OUT/control.log"; then
  echo "  control: PASSED -- the self-test cannot tell the fix from its absence"; verdict=1
else
  echo "  control: the old runner fails the same boot, as it must"
fi
echo "  compiler $("$CAPSTONE_CLANG" --version | grep -oE '[0-9a-f]{40}' | cut -c1-12), logs $OUT"
exit $verdict
