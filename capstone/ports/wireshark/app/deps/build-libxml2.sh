#!/usr/bin/env bash
# libxml2 for capstone64, with its gates:
#   1. native: upstream's own test suite (make check) passes on the host, same configuration;
#   2. cross: libxml2.a builds with capstone-cc and the cast census on, and is installed;
#   3. upstream's xmllint links as a capstone64 domain.
# Without ICU (the census's six libxml2-including files failed only on an ICU header), Python,
# zlib and readline (2.15 has no LZMA option); without threads, as a domain runs one.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch libxml2)
N=$TS_DEPS_BUILD/libxml2-native X=$TS_DEPS_BUILD/libxml2-cap; LOG=$TS_DEPS_BUILD/libxml2-logs
rm -rf "$N" "$X" "$LOG"; mkdir -p "$N" "$X" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1
OPTS=(--disable-shared --enable-static --without-python --without-icu --without-zlib
      --without-readline --without-threads)

# Upstream's check-local runs its test programs one after another and stops at the first failure.
# testModule dlopens a shared test module, which a --disable-shared build does not have ("Failed to
# open module"), so the same commands run here one by one: every one except testModule must exit 0.
( cd "$N" && env -u CC -u AR -u RANLIB ./configure "${OPTS[@]}" > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 > "$LOG/native-build.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -k -j16 check > "$LOG/native-make-check.log" 2>&1 ) || true
: > "$LOG/native-check.log"
( cd "$N" && [ -d test ] || ln -s "$N/test" test ) 2>/dev/null || true
XCHECKS=("./runtest" "./testrecurse" "./testapi" "./testcatalog" "./testchar" "./testdict" "./testparser"
         "./runxmlconf -d xmlconf" "./runsuite" "test/scripts/test.sh ./xmllint"
         "test/catalogs/test.sh ./xmlcatalog" "test/catalogs/test_sgml.sh ./xmlcatalog")
nfail=0
for c in "${XCHECKS[@]}"; do
  if ( cd "$N" && ASAN_OPTIONS=detect_leaks=0 bash -c "$c" ) >> "$LOG/native-check.log" 2>&1; then
    printf '\nPASS: %s\n' "$c" >> "$LOG/native-check.log"   # own line: a test may end without one
  else
    printf '\nFAIL: %s\n' "$c" >> "$LOG/native-check.log"; nfail=$((nfail + 1))
  fi
done
grep -E "^(PASS|FAIL): |^Total [0-9]+ tests" "$LOG/native-check.log" | tr '\n' ' '; echo
[ "$nfail" = 0 ] || { echo "libxml2: NATIVE CHECK FAILED ($nfail)" >&2; exit 1; }
echo "libxml2: native check OK (${#XCHECKS[@]} upstream checks; testModule excluded: static build;"
echo "  runxmlconf runs 0 tests here: the W3C xmlconf suite is not in the release tarball)"

( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --prefix="$TS_DEPS_PREFIX" "${OPTS[@]}" \
    > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 libxml2.la > "$LOG/cap-build.log" 2>&1 )
( cd "$X" && make install-libLTLIBRARIES > "$LOG/cap-install.log" 2>&1 &&
  make -C include install >> "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "libxml2: libxml2.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libxml2.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
( cd "$X" && make xmllint > "$LOG/cap-tool-link.log" 2>&1 )
"$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/xmllint" > /dev/null || { echo "libxml2: xmllint did not link" >&2; exit 1; }
echo "libxml2: xmllint links as a domain ($(stat -c %s "$X/xmllint") bytes)"
