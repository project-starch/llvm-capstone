#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
VERSION=1.37.0
SHA256=3311dff32e746499f4df0d5df04d7eb396382d7e108bb9250e7b519b837043a4
URL=https://busybox.net/downloads/busybox-${VERSION}.tar.bz2
ROOT=$(realpath -m "${1:?usage: prepare-sources.sh SCRATCH_ROOT}")
TMP_ROOT=$(realpath -m "$CAPSTONE_TMP_ROOT")
case "$ROOT/" in "$TMP_ROOT/"*) ;; *) echo "source root must be under CAPSTONE_TMP_ROOT" >&2; exit 2;; esac
TREE=$ROOT/busybox-$VERSION
ARCHIVE=$ROOT/busybox-$VERSION.tar.bz2
PATCH_HASH=$(cat "$HERE"/patches/*.patch | sha256sum | cut -d' ' -f1)
mkdir -p "$ROOT"
if [[ -d $TREE ]]; then
    [[ $(cat "$TREE/.capstone-patches" 2>/dev/null) == "$PATCH_HASH" ]] || {
        echo "source tree differs; choose a fresh scratch root: $TREE" >&2; exit 2;
    }
    printf '%s\n' "$TREE"
    exit 0
fi
if [[ ! -f $ARCHIVE ]]; then
    curl --fail --location --silent --show-error "$URL" -o "$ARCHIVE.part"
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | sha256sum --check --status
tar -xjf "$ARCHIVE" -C "$ROOT"
for patch_file in "$HERE"/patches/*.patch; do
    patch --directory="$TREE" --strip=1 --forward --silent < "$patch_file"
done
printf '%s\n' "$PATCH_HASH" > "$TREE/.capstone-patches"
printf '%s\n' "$TREE"
