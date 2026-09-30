#!/usr/bin/env bash
# Verify the bring-up. Run through run.sh:  ./run.sh capstone/container/verify.sh
#
# Checks run in dependency order and each one REPORTS rather than aborting, so a single
# failure does not hide the state of everything after it. Exit status is non-zero if any
# check failed.
#
# Checks 6 and 7 boot QEMU. They take $CAPSTONE_QEMU_LOCK, and the project rule is that
# QEMU suites never run two at a time (shared rootfs.ext2).
set -uo pipefail

ROOT=${CAPSTONE_REPO_ROOT:?must run through run.sh}
cd "$ROOT"
FAILED=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILED=1; }
skip() { printf '  skip  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

head_ "1. image toolchain"
if out=$(gcc --version 2>&1 | head -1); then pass "$out"; else fail "gcc missing"; fi
if out=$(cmake --version 2>&1 | head -1); then pass "$out"; else fail "cmake missing"; fi
if out=$(ninja --version 2>&1); then pass "ninja $out"; else fail "ninja missing"; fi
if out=$(python3 --version 2>&1); then pass "$out"; else fail "python3 missing"; fi

head_ "2. env contract resolves (capstone-test-env.sh must exit 0)"
# The likeliest silent failure in the whole setup. The script validates that the root has
# llvm/ and capstone/ under it and returns 1 otherwise; a stale build dir is a WARNING on
# stderr, not a failure, so 0 here still means the paths are right.
# stderr is captured separately: capstone-test-env.sh's toolchain-fresh warning goes to
# stderr and is NOT a failure, so mixing the two streams makes a healthy run look broken.
# (rc=1 "stale" is normal here -- setup.sh builds a named subset of targets, so ninja
# always still has work queued for the ones we never ask for.)
if out=$(bash -c 'source capstone/tests/capstone-test-env.sh >/dev/null 2>/dev/null && \
        printf "%s\n%s\n%s\n" "$CAPSTONE_REPO_ROOT" "$CAPSTONE_LLVM_BIN" "$CAPSTONE_QEMU_BINARY"'); then
  pass "sourced clean (rc=0)"
  printf '%s\n' "$out" | sed 's/^/        /'
else
  fail "capstone-test-env.sh returned non-zero:"
  printf '%s\n' "$out" | sed 's/^/        /'
fi

# Everything below needs the env. Source it once; keep going even if it warned.
set +e
source capstone/tests/capstone-test-env.sh 2>/dev/null
set -e 2>/dev/null || true

head_ "3. toolchain binaries (ONBOARDING section 3)"
if [ -x "${CAPSTONE_CLANG:-}" ]; then pass "$("$CAPSTONE_CLANG" --version | head -1)"
else fail "no clang at ${CAPSTONE_CLANG:-unset}"; fi
if [ -x "${CAPSTONE_LD_LLD:-}" ]; then pass "$("$CAPSTONE_LD_LLD" --version | head -1)"
else fail "no ld.lld at ${CAPSTONE_LD_LLD:-unset}"; fi
if [ -x "${CAPSTONE_QEMU_BINARY:-}" ]; then pass "$("$CAPSTONE_QEMU_BINARY" --version | head -1)"
else fail "no qemu at ${CAPSTONE_QEMU_BINARY:-unset}"; fi

head_ "4. the Capstone target is really registered"
# A clang built with the target accidentally dropped still prints a version banner and
# would pass check 3. Ask llc what it registered.
if [ -x "${CAPSTONE_LLVM_BIN:-}/llc" ]; then
  if "$CAPSTONE_LLVM_BIN/llc" --version 2>&1 | grep -qi capstone; then
    pass "llc lists the capstone target"
  else
    fail "llc does NOT list a capstone target"
  fi
else
  fail "no llc at ${CAPSTONE_LLVM_BIN:-unset}/llc"
fi

head_ "5. codegen end to end (my_first_domain)"
# build.sh defaults LLVM_BIN to llvm/build/bin, which is NOT where we build. Pass it.
if [ -x "${CAPSTONE_CLANG:-}" ]; then
  if out=$(cd capstone/my_first_domain && LLVM_BIN="$CAPSTONE_LLVM_BIN" bash build.sh 2>&1); then
    if [ -f capstone/my_first_domain/my_domain.dom ]; then pass "my_domain.dom produced"
    else fail "build.sh exited 0 but produced no my_domain.dom"; fi
  else
    fail "build.sh failed:"; printf '        %s\n' "$(printf '%s' "$out" | tail -15)"
  fi
else
  skip "no clang yet"
fi

head_ "5b. the LLVM-built domain actually RUNS on capstone-qemu"
# Check 5 only proves my_domain.dom was PRODUCED. That is a different fact from "a domain
# built by this toolchain executes", and conflating the two hides exactly the failures worth
# catching: a domain that links cleanly and then faults, or a link.ld/start.S ABI drift that
# only shows at domreturn. So build it with OUR toolchain and boot it.
#
# build.sh defaults LLVM_BIN to llvm/build/bin, which is NOT this build tree -- without the
# override it can silently compile with some other clang, or fail for the wrong reason.
if [ -f "${CAPSTONE_BUILDROOT_DIR:-}/build/images/Image" ] && [ -x "${CAPSTONE_QEMU_BINARY:-}" ]; then
  MFD_SHARE="${CAPSTONE_TMP_ROOT:-/tmp/capstone}/my-first-domain-share"
  MFD_LOG="${CAPSTONE_TMP_ROOT:-/tmp/capstone}/my-first-domain.log"
  if out=$( set -e
            rm -rf "$MFD_SHARE"; mkdir -p "$MFD_SHARE"
            ( cd capstone/my_first_domain && LLVM_BIN="$CAPSTONE_LLVM_BIN" bash build.sh )
            cp capstone/my_first_domain/my_domain.dom "$MFD_SHARE/"
            bash capstone/tests/runtime-qemu/build-capstone-test-user.sh "$MFD_SHARE/capstone-test.user"
            # $CAPSTONE_QEMU_LOCK, not a private one: the suites share rootfs.ext2 and the
            # project rule is that no two QEMU runs overlap.
            flock "$CAPSTONE_QEMU_LOCK" python3 capstone/tests/runtime-qemu/run-domain-smoke.py \
              --share-dir "$MFD_SHARE" --log-file "$MFD_LOG" \
              --guest-command "/mnt/host/capstone-test.user /mnt/host/my_domain.dom" \
              --success-marker "Created domain ID = 0" \
              --success-marker "Called dom (1-th time) retval = 42" ) 2>&1; then
    # Confirm against the serial log, not the wrapper's exit status.
    if grep -qF 'retval = 42' "$MFD_LOG" 2>/dev/null; then
      pass "my_domain.dom (clang -target capstone64 + ld.lld) ran in the guest, retval = 42"
    else
      fail "wrapper exited 0 but 'retval = 42' is absent from $MFD_LOG"
    fi
  else
    fail "my_domain.dom did not run:"; printf '%s\n' "$out" | tail -20 | sed 's/^/        /'
  fi
else
  skip "needs both the guest image and qemu"
fi

head_ "6. guest boots (run-smoke.sh)"
# The markers are asserted against the SERIAL LOG, not run-smoke.sh's stdout. stdout only
# carries "QEMU smoke passed."; the guest console goes to $LOG_FILE. Grepping stdout for
# "retval = 42" reports FAIL on a run that actually passed.
SMOKE_LOG="${CAPSTONE_TMP_ROOT:-/tmp/capstone}/capstone-runtime-qemu-smoke.log"
if [ -f "${CAPSTONE_BUILDROOT_DIR:-}/build/images/Image" ] && [ -x "${CAPSTONE_QEMU_BINARY:-}" ]; then
  if out=$(bash capstone/tests/runtime-qemu/run-smoke.sh 2>&1); then
    miss=0
    for m in 'Created domain ID = 0' 'Called dom (1-th time) retval = 42'; do
      grep -qF "$m" "$SMOKE_LOG" 2>/dev/null || { fail "marker absent from serial log: $m"; miss=1; }
    done
    [ "$miss" = 0 ] && pass "guest booted, domain created, retval = 42"
  else
    fail "run-smoke.sh failed:"; printf '%s\n' "$out" | tail -20 | sed 's/^/        /'
  fi
else
  skip "needs both the guest image and qemu"
fi

head_ "7. QEMU probe suites (serialized by \$CAPSTONE_QEMU_LOCK)"
for s in capstone/capstone-qemu/tests/capstone-revoke-probes/run-revoke-probes.sh \
         capstone/capstone-qemu/tests/capstone-mrev-codegen/run-mrev-codegen-probes.sh; do
  if [ ! -f "$s" ]; then skip "$s (not at this QEMU pin)"; continue; fi
  if [ ! -f "${CAPSTONE_BUILDROOT_DIR:-}/build/images/Image" ]; then skip "$(basename "$s") (no guest image)"; continue; fi
  if out=$(bash "$s" 2>&1); then pass "$(basename "$s")"
  else fail "$(basename "$s"):"; printf '        %s\n' "$(printf '%s' "$out" | tail -20)"; fi
done

head_ "8. host ownership of build outputs"
# If --userns=keep-id is not taking effect, these come back owned by subuid 101000 and
# the tree cannot be deleted or edited from the host without `podman unshare`.
for p in "${CAPSTONE_LLVM_BUILD_DIR:-}/bin/clang" capstone/capstone-qemu/build/qemu-system-riscv64; do
  [ -e "$p" ] || { skip "$p not built"; continue; }
  owner=$(stat -c '%u' "$p")
  if [ "$owner" = "$(id -u)" ]; then pass "$p owned by uid $owner"
  else fail "$p owned by uid $owner, expected $(id -u) -- keep-id is not in effect"; fi
done

printf '\n'
[ "$FAILED" = 0 ] && { printf '\033[32mall checks passed\033[0m\n'; exit 0; }
printf '\033[31msome checks failed (see above)\033[0m\n'; exit 1
