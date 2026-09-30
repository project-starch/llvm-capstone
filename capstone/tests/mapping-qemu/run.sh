#!/usr/bin/env bash
# Bare-metal Capstone tests on capstone-qemu. Each tests/*.S declares
#   // EXPECT: <exit code>
# and reports through virt's test device. Exit status: number of mismatches.
# Usage: run.sh [-q QEMU] [test.S ...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
QEMU=${QEMU:-${CAPSTONE_QEMU_BINARY:-$HERE/../../capstone-qemu/build/qemu-system-riscv64}}
CLANG=${CLANG:-/usr/bin/clang}
while getopts q: opt; do case $opt in q) QEMU=$OPTARG;; esac; done
shift $((OPTIND - 1))
OUT=${OUT:-$(mktemp -d /tmp/mapping-qemu.XXXXXX)}
tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=("$HERE"/tests/*.S)
fail=0; n=0
for src in "${tests[@]}"; do
  name=$(basename "$src" .S); n=$((n + 1))
  expect=$(sed -nE 's#^// EXPECT: *([0-9a-fx]+).*#\1#p' "$src" | head -1)
  [ -n "$expect" ] || { echo "MISSING-EXPECT $name"; fail=$((fail + 1)); continue; }
  expect=$((expect))
  if ! "$CLANG" -target riscv64-unknown-elf -march=rv64imac_zicsr -mabi=lp64 -nostdlib \
        -fuse-ld=lld -Wl,-T,"$HERE/link.ld" -I"$HERE" -o "$OUT/$name.elf" "$src" \
        > "$OUT/$name.build.log" 2>&1; then
    echo "BUILD-FAIL $name (see $OUT/$name.build.log)"; fail=$((fail + 1)); continue
  fi
  timeout 30 "$QEMU" -M virt -smp 1 -m 256M -nographic -bios none \
    -kernel "$OUT/$name.elf" > "$OUT/$name.run.log" 2>&1
  rc=$?
  if [ "$rc" -eq "$expect" ]; then
    printf 'PASS %-40s exit %3d\n' "$name" "$rc"
  else
    printf 'FAIL %-40s exit %3d expected %3d (log: %s)\n' "$name" "$rc" "$expect" "$OUT/$name.run.log"
    fail=$((fail + 1))
  fi
done
echo "$((n - fail))/$n passed; qemu=$QEMU; out=$OUT"
exit $fail
