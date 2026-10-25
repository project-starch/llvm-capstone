#!/usr/bin/env bash
# Fetch the NIST SARD Juliet C/C++ 1.3 test suite (public domain) and unpack it, refusing any archive whose
# sha256 differs from the pinned one. Juliet itself is never committed here. The tree is always extracted afresh
# from the verified archive, so a cached, edited or half-extracted tree is never taken as the source.
#
#   fetch-juliet.sh <dest-dir>      prints <dest-dir>/C, which holds testcases/ and testcasesupport/
set -euo pipefail
dest=${1:?destination directory}
url=https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
sha=ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)
zip=$dest/juliet-1.3.zip
if [ ! -f "$zip" ]; then
    curl -fsSL -o "$zip.part" "$url"
    mv "$zip.part" "$zip"
fi
echo "$sha  $zip" | sha256sum -c --quiet || { echo "juliet archive hash mismatch: $zip" >&2; exit 1; }
tmp=$(mktemp -d "$dest/extract.XXXXXX")
unzip -q "$zip" -d "$tmp"
test -f "$tmp/C/testcasesupport/io.c"
rm -rf "$dest/C"
mv "$tmp/C" "$dest/C"
rm -rf "$tmp"
echo "$dest/C"
