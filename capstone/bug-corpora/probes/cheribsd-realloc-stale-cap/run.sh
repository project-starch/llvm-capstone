#!/usr/bin/env bash
# Build probe.c for riscv64-purecap and run it in one CheriBSD boot, revocation ON by default.
# The probe itself re-executes with _RUNTIME_REVOCATION_DISABLE=1 for the jemalloc-only arm.
#
#   CHERI_SDK=~/cheri/output/sdk CHERI_SYSROOT=~/cheri/output/rootfs-riscv64-purecap \
#   CHERI_IMAGE=~/cheri/output/cheribsd-riscv64-purecap.img bash run.sh <fresh outdir>
#
# Controls, all from the same boot: the ABI probe (pointer_bytes=16, runtime_revocation=1), the
# bounds probe, and revocation-control (a forced sweep must clear a tag and the read must fault).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd "$HERE/../../.." && pwd)
SDK=${CHERI_SDK:-$HOME/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheri/output/rootfs-riscv64-purecap}
IMAGE=${CHERI_IMAGE:-$HOME/cheri/output/cheribsd-riscv64-purecap.img}
PORT=${CHERI_PORT:-10479}
# run.py imports pexpect; the system python3 lacks it, the project venv has it.
PYTHON=${PYTHON:-$HOME/.venvs/capstone/bin/python3}
OUT=${1:?usage: run.sh <fresh outdir>}
"$PYTHON" -c 'import pexpect' 2>/dev/null || { echo "CONTROL-FAILED $PYTHON has no pexpect" >&2; exit 75; }
for p in "$SDK/bin/clang" "$SYSROOT" "$IMAGE"; do
  [ -e "$p" ] || { echo "CONTROL-FAILED missing $p" >&2; exit 75; }
done
[ -e "$OUT/run" ] && { echo "CONTROL-FAILED $OUT/run exists" >&2; exit 75; }
mkdir -p "$OUT/bin"
CFLAGS="--target=riscv64-unknown-freebsd13 -march=rv64imafdcxcheri -mabi=l64pc128d"
CFLAGS="$CFLAGS -mno-relax -B$SDK/bin --sysroot=$SYSROOT -std=gnu11 -O0 -fuse-ld=lld"
"$SDK/bin/clang" $CFLAGS "$HERE/probe.c" -o "$OUT/bin/realloc-probe" || exit 75
"$SDK/bin/clang" $CFLAGS "$CAP/ports/common/host/cheribsd/abi-probe.c" -o "$OUT/bin/cheribsd-abi-probe" || exit 75
"$SDK/bin/clang" $CFLAGS "$CAP/ports/memcached/allocators/security-tests/cheribsd/revocation-control.c" \
  -o "$OUT/bin/revocation-control" || exit 75
cat > "$OUT/cases.json" <<EOF
[
  {"name": "revocation-control", "program": "$OUT/bin/revocation-control",
   "expect_regex": "REVOCATION_CONTROL revocation=1 tag_after_sweep=0 reissued=[01]", "exit": 162},
  {"name": "realloc-probe", "program": "$OUT/bin/realloc-probe", "expect": "PROBE_DONE", "exit": 0}
]
EOF
"$PYTHON" "$CAP/ports/common/host/cheribsd/run.py" "$OUT/run" \
  --sdk "$SDK" --rootfs "$SYSROOT" --image "$IMAGE" --port "$PORT" \
  --abi-probe "$OUT/bin/cheribsd-abi-probe" --runtime-revocation on \
  --cases "$OUT/cases.json" --continue-on-failure
rc=$?
echo "run.sh: suite exit $rc; per-case output in $OUT/run/"
exit "$rc"
