#!/usr/bin/env bash
# Native ABI keeps mallocng's group layout and all policy sources unchanged.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../tests/capstone-test-env.sh"
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
out=${1:?Output directory}
out=$(mkdir -p "$out"; cd "$out"; pwd)
src=$(MUSL_CACHE_ROOT="$out/source" bash "$here/../../ports/musl-capstone/fetch-musl.sh" | tail -1)
python3 "$here/../../ports/musl-capstone/check-mallocng-policy.py" "$src" "$out/source/musl-1.2.5.tar.gz" --include-glue >&2
mkdir -p "$out/build"
cd "$out/build"
if [[ ! -f config.mak ]]; then
  CC="${CROSS_COMPILE}gcc" "$src/configure" --target=riscv64 --disable-shared --prefix="$out/sysroot" >&2
fi
make -j"${JOBS:-16}" >&2
make install >&2
printf '%s\n' "$out/sysroot"
