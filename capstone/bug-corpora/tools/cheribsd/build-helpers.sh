#!/usr/bin/env bash
# build-helpers.sh OUT: the CheriBSD purecap helpers a corpus runner pushes into the guest.
#   sicode.so             LD_PRELOAD: one SICODE line naming the fault that ended the process
#   sicode-selftest       its positive control, run with sicode.so preloaded (bounds, revoked)
#   quarantine-probe.so   LD_PRELOAD: whether each free entered MRS quarantine, whether memory came
#                         back while still quarantined, and how many sweeps ran
# The flags are the SDK's own purecap configuration (bin/cheribsd-riscv64-purecap.cfg) with the
# sysroot named explicitly.
set -euo pipefail
: "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the purecap sysroot}"
[[ $# == 1 ]] || { echo "usage: build-helpers.sh OUT" >&2; exit 2; }
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROBE=$HERE/../../postgres/mmgr-repros/results/20261008-cheribsd/quarantine-probe.c
OUT=$1
mkdir -p "$OUT"
CC=("$CHERI_SDK/bin/clang" --target=riscv64-unknown-freebsd13 --sysroot="$CHERI_SYSROOT"
    -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -cheri-tgot-tls -fuse-ld=lld
    -B"$CHERI_SDK/bin" -O1 -Wall)
"${CC[@]}" -shared -fPIC "$HERE/sicode.c" -o "$OUT/sicode.so"
"${CC[@]}" -DSICODE_SELFTEST "$HERE/sicode.c" -o "$OUT/sicode-selftest"
"${CC[@]}" -shared -fPIC "$PROBE" -o "$OUT/quarantine-probe.so"
( cd "$OUT" && sha256sum sicode.so sicode-selftest quarantine-probe.so > SHA256SUMS )
cat "$OUT/SHA256SUMS"
