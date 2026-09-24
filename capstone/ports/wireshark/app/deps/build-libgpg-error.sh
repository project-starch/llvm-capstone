#!/usr/bin/env bash
# libgpg-error for capstone64, with its gates:
#   1. native: upstream's own test suite (make check) passes on the host, same configuration;
#   2. cross: libgpg-error.a builds with capstone-cc and the cast census on, and is installed;
#   3. upstream's gpg-error tool links as a capstone64 domain.
# --disable-threads: a domain runs one thread (tshark.c:1370 registers inline), and it spares
# the per-target lock-object header, whose layout depends on pthread_mutex_t's size.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch libgpg-error)
N=$TS_DEPS_BUILD/libgpg-error-native X=$TS_DEPS_BUILD/libgpg-error-cap; LOG=$TS_DEPS_BUILD/libgpg-error-logs
rm -rf "$N" "$X" "$LOG"; mkdir -p "$N" "$X" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1
OPTS=(--disable-shared --enable-static --disable-threads --disable-nls --disable-doc --disable-languages)

# Upstream's tests/t-poll.c does not compile with --disable-threads (it names producer_thread and
# consumer_thread unconditionally, t-poll.c:441-442), and automake builds every test program before
# running any. So the suite runs with t-poll taken out of BOTH lists; every other test must pass.
GPGE_TESTS="t-version t-strerror t-syserror t-lock t-printf t-b64 t-argparse t-logging t-stringutils t-malloc t-spawn t-strlist t-name-value"
# The native build is also installed into a native-only prefix: libgcrypt's native check links it.
NP=$TS_DEPS_BUILD/native-prefix
( cd "$N" && env -u CC -u AR -u RANLIB ./configure --prefix="$NP" "${OPTS[@]}" > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 -C src > "$LOG/native-build.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -C tests $GPGE_TESTS > "$LOG/native-tests-build.log" 2>&1 ) || true
: > "$LOG/native-check.log"
for tst in $GPGE_TESTS; do   # each test program run directly; exit status 0 is a pass, as automake counts it
  if [ -x "$N/tests/$tst" ] && ( cd "$N/tests" && ./$tst ) >> "$LOG/native-check.log" 2>&1; then
    echo "PASS: $tst" >> "$LOG/native-check.log"
  else
    echo "FAIL: $tst" >> "$LOG/native-check.log"
  fi
done
npass=$(grep -c '^PASS:' "$LOG/native-check.log" || true)
nwant=$(echo $GPGE_TESTS | wc -w)
if grep -qE "^FAIL:|^ERROR:" "$LOG/native-check.log" || [ "$npass" -ne "$nwant" ]; then
  echo "libgpg-error: NATIVE CHECK FAILED (PASS $npass of $nwant)" >&2; exit 1
fi
( cd "$N" && env -u CC -u AR -u RANLIB make -C src install > "$LOG/native-install.log" 2>&1 )
echo "libgpg-error: native check OK ($npass of $nwant tests PASS; t-poll not built: upstream defect under --disable-threads)"

( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --prefix="$TS_DEPS_PREFIX" "${OPTS[@]}" \
    > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 -C src > "$LOG/cap-build.log" 2>&1 )
( cd "$X" && make -C src install > "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "libgpg-error: libgpg-error.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libgpg-error.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
"$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/src/gpg-error" > /dev/null || { echo "libgpg-error: gpg-error did not link" >&2; exit 1; }
echo "libgpg-error: gpg-error links as a domain ($(stat -c %s "$X/src/gpg-error") bytes)"
