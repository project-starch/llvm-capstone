#!/usr/bin/env bash
# c-ares for capstone64, with its gates:
#   0. the port's patches (deps/patches/c-ares-*) are applied to BOTH trees, so the native suite
#      tests the patched code;
#   1. native: upstream's own test suite (arestest, the mock-server tests; the Live* tests need
#      a network and are excluded) passes on the host, built against a pinned googletest;
#   2. cross: libcares.a builds with capstone-cc (autotools, static) and the cast census on;
#   3. upstream's adig tool links as a capstone64 domain.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch c-ares); GT=$(ts_fetch googletest)
N=$TS_DEPS_BUILD/c-ares-native X=$TS_DEPS_BUILD/c-ares-cap G=$TS_DEPS_BUILD/googletest-native
LOG=$TS_DEPS_BUILD/c-ares-logs
rm -rf "$N" "$X" "$G" "$LOG"; mkdir -p "$N" "$X" "$G/src" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1
tar -xf "$GT" -C "$G/src" --strip-components=1
for d in "$N" "$X"; do
  for p in "$TS_DEPS_DIR"/patches/c-ares-*.patch; do patch -s -d "$d" -p1 < "$p"; done
done

NATIVE_ENV=(env -u CC -u AR -u RANLIB)
"${NATIVE_ENV[@]}" cmake -S "$G/src" -B "$G/build" -DCMAKE_INSTALL_PREFIX="$G/install" -DBUILD_GMOCK=ON \
  > "$LOG/gtest-cmake.log" 2>&1
"${NATIVE_ENV[@]}" cmake --build "$G/build" -j16 --target install > "$LOG/gtest-build.log" 2>&1
"${NATIVE_ENV[@]}" cmake -S "$N" -B "$N/build" -DCARES_BUILD_TESTS=ON -DCARES_STATIC=ON -DCARES_SHARED=OFF \
  -DCMAKE_PREFIX_PATH="$G/install" > "$LOG/native-cmake.log" 2>&1
"${NATIVE_ENV[@]}" cmake --build "$N/build" -j16 > "$LOG/native-build.log" 2>&1
"$N/build/bin/arestest" --gtest_filter='-*.Live*' > "$LOG/native-test.log" 2>&1 || true
tail -3 "$LOG/native-test.log"
grep -qE "^\[  PASSED  \] [0-9]+ tests" "$LOG/native-test.log" && ! grep -q "^\[  FAILED  \]" "$LOG/native-test.log" \
  || { echo "c-ares: NATIVE TEST FAILED" >&2; exit 1; }
echo "c-ares: native test OK"

( cd "$X" && ./configure --host=riscv64-unknown-linux-musl --prefix="$TS_DEPS_PREFIX" --disable-shared \
    --enable-static --disable-tests > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 -C src/lib > "$LOG/cap-build.log" 2>&1 )
( cd "$X" && make -C src/lib install > "$LOG/cap-install.log" 2>&1 && make -C include install >> "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "c-ares: libcares.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libcares.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
( cd "$X" && make -C src/tools adig > "$LOG/cap-tool-link.log" 2>&1 )
"$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/src/tools/adig" > /dev/null || { echo "c-ares: adig did not link" >&2; exit 1; }
echo "c-ares: adig links as a domain ($(stat -c %s "$X/src/tools/adig") bytes)"
