#!/usr/bin/env bash
# Build every microbenchmark image of the virtual profile for ONE arm.
#   build-all.sh <sdk> <out>          measured images: <out>/<name>.dom, <out>/images.tsv
# Variants of the same tree (the SDK argument is ignored for native):
#   BENCH_MODE=count  build-all.sh <sdk> <out>   counting images, QEMU functional pass ONLY
#   BENCH_MODE=native build-all.sh -     <out>   riscv64 gcc -O2 -static references
# <sdk> is the SDK's directory holding capstone-cc and sdk.json (e.g. .../sdk-T1/sdk).
# Environment: CAPSTONE_LLVM_BIN (the SDK's compiler), JOBS (<= 12; default 12).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sdk=${1:?usage: build-all.sh <sdk> <out>}
out=${2:?usage: build-all.sh <sdk> <out>}
mode=${BENCH_MODE:-capstone}
args=(--mode "$mode" --out "$out" --jobs "${JOBS:-12}")
[[ $mode == native ]] || args+=(--sdk "$sdk")
unset CAPSTONE_CLANG CCC_OVERRIDE_OPTIONS
exec python3 "$here/build.py" "${args[@]}"
