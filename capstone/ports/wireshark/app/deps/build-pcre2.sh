#!/usr/bin/env bash
# PCRE2 (8-bit, no JIT) for capstone64, with its gates:
#   1. native: upstream's own test suite (make check) passes on the host;
#   2. cross: libpcre2-8.a builds with capstone-cc and the cast census on, and is installed;
#   3. upstream's pcre2test links as a capstone64 domain.
# JIT is off: it generates machine code at run time, which a domain cannot map executable.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch pcre2)
N=$TS_DEPS_BUILD/pcre2-native X=$TS_DEPS_BUILD/pcre2-cap; LOG=$TS_DEPS_BUILD/pcre2-logs
rm -rf "$N" "$X" "$LOG"; mkdir -p "$N" "$X" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1
OPTS=(--disable-shared --enable-static --disable-jit --enable-pcre2-8 --disable-pcre2-16
      --disable-pcre2-32 --disable-pcre2grep-libz --disable-pcre2grep-libbz2
      --disable-pcre2test-libedit --disable-pcre2test-libreadline)

( cd "$N" && env -u CC -u AR -u RANLIB ./configure "${OPTS[@]}" > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 > "$LOG/native-build.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make check -j16 > "$LOG/native-check.log" 2>&1 ) || true
grep -E "^# (TOTAL|PASS|FAIL|ERROR):" "$LOG/native-check.log" | tr '\n' ' '; echo
grep -qE "^# FAIL: +0$" "$LOG/native-check.log" && grep -qE "^# ERROR: +0$" "$LOG/native-check.log" \
  || { echo "pcre2: NATIVE CHECK FAILED" >&2; exit 1; }
echo "pcre2: native check OK"

( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --prefix="$TS_DEPS_PREFIX" "${OPTS[@]}" \
    > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 libpcre2-8.la > "$LOG/cap-build.log" 2>&1 )
( cd "$X" && make install-libLTLIBRARIES install-includeHEADERS install-nodist_includeHEADERS \
    > "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "pcre2: libpcre2-8.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libpcre2-8.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
( cd "$X" && make pcre2test > "$LOG/cap-test-link.log" 2>&1 )
"$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/pcre2test" > /dev/null || { echo "pcre2: pcre2test did not link" >&2; exit 1; }
echo "pcre2: pcre2test links as a domain ($(stat -c %s "$X/pcre2test") bytes)"
