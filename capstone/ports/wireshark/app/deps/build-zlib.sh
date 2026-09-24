#!/usr/bin/env bash
# zlib for capstone64, with its gates:
#   1. native: upstream's own test suite (make test) passes on the host;
#   2. cross: libz.a builds with capstone-cc, the cast census on (TS_CENSUS=1), and is installed
#      into $TS_DEPS_PREFIX; every lossy/rebuilt cast site is listed for classification;
#   3. upstream's test programs (example, minigzip) LINK as capstone64 domains.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch zlib)
N=$TS_DEPS_BUILD/zlib-native X=$TS_DEPS_BUILD/zlib-cap; LOG=$TS_DEPS_BUILD/zlib-logs
rm -rf "$N" "$X" "$LOG"; mkdir -p "$N" "$X" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1

( cd "$N" && env -u CC -u AR -u RANLIB ./configure --static > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 test > "$LOG/native-test.log" 2>&1 )
grep -q '\*\*\* zlib test OK \*\*\*' "$LOG/native-test.log" || { echo "zlib: NATIVE TEST FAILED" >&2; exit 1; }
echo "zlib: native test OK"

( cd "$X" && CHOST=riscv64-capstone ./configure --static --prefix="$TS_DEPS_PREFIX" > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 libz.a > "$LOG/cap-build.log" 2>&1 && make install > "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "zlib: libz.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libz.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
( cd "$X" && make example minigzip > "$LOG/cap-tests-link.log" 2>&1 )
for t in example minigzip; do
  "$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/$t" | grep -q "Machine:.*" || { echo "zlib: $t did not link" >&2; exit 1; }
done
echo "zlib: example and minigzip link as domains: $(file -b "$X/example" | cut -c1-60)"
