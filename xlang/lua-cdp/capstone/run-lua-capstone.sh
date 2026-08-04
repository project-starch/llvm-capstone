#!/usr/bin/env bash
# Run the Lua-CDP Capstone column: each row, both variants, ONE boot each.
#
#   ./run-lua-capstone.sh                      # every shim under shims/
#   ./run-lua-capstone.sh luac_openssl_ctx_uaf # one row
#
# ONE BOOT PER RUN, deliberately: a faulting domain halts QEMU, so every row
# after it in the same boot silently never runs. Builds the host controller
# (reused from ../../capstone) and both domain variants, then runs them.
#
# Reuses the mruby column's host (xlang_shim_host.c) and libcapstone verbatim;
# the domains are built by ./build-lua-capstone.sh.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CAPSTONE_REPO_ROOT:=$(cd "$HERE/../../.." && pwd)}"
export CAPSTONE_REPO_ROOT
# shellcheck source=/dev/null
source "$CAPSTONE_REPO_ROOT/capstone/tests/capstone-test-env.sh"

MRUBY_DIR="$CAPSTONE_REPO_ROOT/xlang/capstone"
SHARE=${SHARE:-$CAPSTONE_TMP_ROOT/lua-cdp-capstone}
SMOKE="$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py"
RESULTS="$SHARE/results.tsv"
ATTEMPTS=${ATTEMPTS:-5}
mkdir -p "$SHARE"

rows=("$@")
if [ ${#rows[@]} -eq 0 ]; then
  rows=()
  for s in "$HERE"/../shims/*.c; do
    b=$(basename "$s" .c)
    [ "$b" = luac_shim ] && continue   # the header-companion, not a row
    rows+=("$b")
  done
fi

# --- build host controller (once) ---------------------------------------------
GUEST_CC=${GUEST_CC:-$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc}
LIBCAPSTONE="$CAPSTONE_REPO_ROOT/capstone/caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.c"
"$GUEST_CC" -O2 -I"$MRUBY_DIR" -I"$CAPSTONE_REPO_ROOT/capstone/benchmarks/sqlite" \
  -o "$SHARE/xlang_host.user" "$MRUBY_DIR/xlang_shim_host.c" "$LIBCAPSTONE" \
  || { echo "HOST BUILD FAILED" >&2; exit 1; }

classify() { # $1 = log file
  local log="$1"
  grep -qa 'XLANG-INVALID'                     "$log" && { echo "INVALID	harness could not measure"; return; }
  grep -qa 'survived'                          "$log" && { echo "MISS	stale access completed"; return; }
  grep -qa 'domain halted by capability fault' "$log" && { echo "FAULT	$(grep -oa 'cause = [0-9]*' "$log" | tail -1)"; return; }
  grep -qa "Assertion \`rs1_v->tag' failed"    "$log" && { echo "FAULT	qemu assert on untagged operand (emulator gap)"; return; }
  echo "NORESULT	no marker (boot flake or timeout)"
}

printf 'row\tvariant\toutcome\tdetail\n' > "$RESULTS"
for row in "${rows[@]}"; do
  OUT_DIR="$SHARE" ROW="$row" bash "$HERE/build-lua-capstone.sh" >/dev/null 2>&1 \
    || { echo "BUILD FAILED: $row (revoke)" >&2; exit 1; }
  OUT_DIR="$SHARE" ROW="$row" NO_REVOKE=1 bash "$HERE/build-lua-capstone.sh" >/dev/null 2>&1 \
    || { echo "BUILD FAILED: $row (control)" >&2; exit 1; }

  for variant in revoke control; do
    if [ "$variant" = control ]; then dom="xlang_${row}_norevoke"; else dom="xlang_${row}"; fi
    log="$SHARE/run_${row}_${variant}.log"
    outcome="NORESULT	no marker"
    for _ in $(seq 1 "$ATTEMPTS"); do
      timeout 520 python3 "$SMOKE" \
        --share-dir "$SHARE" --log-file "$log" --timeout-multiplier 8 \
        --guest-command "cp /mnt/host/xlang_host.user /tmp/h && chmod 0755 /tmp/h && /tmp/h /mnt/host/$dom.dom" \
        --success-marker '__never__' >/dev/null 2>&1
      outcome=$(classify "$log")
      [ "${outcome%%$'\t'*}" != NORESULT ] && break
    done
    printf '%s\t%s\t%s\n' "$row" "$variant" "$outcome" >> "$RESULTS"
    printf '%-28s %-8s %s\n' "$row" "$variant" "$outcome"
  done
done

echo
echo "results: $RESULTS"
