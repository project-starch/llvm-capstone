#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
sdk=${1:?prepared virtual SDK directory}
out=${2:?output directory, shared with the test guest}
: "${CROSS_COMPILE:?RISC-V Linux compiler prefix}"
mkdir -p "$out"
for name in torture reuse; do
    "$sdk/capstone-cc" -O2 "$here/$name.c" -o "$out/$name"
done
for name in pfns hog; do
    "${CROSS_COMPILE}gcc" -O2 -static "$here/$name.c" -o "$out/$name"
done
sha256sum "$out"/{torture,reuse,pfns,hog} > "$out/inputs.sha256"
