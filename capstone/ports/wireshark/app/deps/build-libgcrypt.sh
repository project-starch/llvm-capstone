#!/usr/bin/env bash
# libgcrypt for capstone64 (after build-libgpg-error.sh), with its gates:
#   0. the port's patches (deps/patches/libgcrypt-*) are applied to BOTH trees;
#   1. native: upstream's own test suite (make check) passes on the host, same configuration;
#   2. cross: libgcrypt.a builds with capstone-cc (no assembly, CAPSTONE_SINGLE_THREAD_DOMAIN)
#      and the cast census on, and is installed;
#   3. upstream's tools (dumpsexp, hmac256, mpicalc) link as capstone64 domains.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
[ -f "$TS_DEPS_PREFIX/lib/libgpg-error.a" ] || { echo "libgcrypt: run build-libgpg-error.sh first" >&2; exit 2; }
TAR=$(ts_fetch libgcrypt)
N=$TS_DEPS_BUILD/libgcrypt-native X=$TS_DEPS_BUILD/libgcrypt-cap; LOG=$TS_DEPS_BUILD/libgcrypt-logs
NP=$TS_DEPS_BUILD/native-prefix
[ -x "$NP/bin/gpgrt-config" ] || { echo "libgcrypt: no native libgpg-error in $NP; run build-libgpg-error.sh" >&2; exit 2; }
rm -rf "$N" "$X" "$LOG"; mkdir -p "$N" "$X" "$LOG"
tar -xf "$TAR" -C "$N" --strip-components=1
tar -xf "$TAR" -C "$X" --strip-components=1
for d in "$N" "$X"; do
  for p in "$TS_DEPS_DIR"/patches/libgcrypt-*.patch; do patch -s -d "$d" -p1 < "$p"; done
done
OPTS=(--disable-shared --enable-static --disable-asm --disable-jent-support --disable-doc)

# Native, against the NATIVE libgpg-error the previous recipe installed (same version, same options).
# configure finds gpgrt-config only on PATH (1.61 installs no gpg-error-config), so each side puts
# its own prefix's bin first.
( cd "$N" && env -u CC -u AR -u RANLIB PATH="$NP/bin:$PATH" ./configure "${OPTS[@]}" --with-libgpg-error-prefix="$NP" \
    > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 > "$LOG/native-build.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j16 check > "$LOG/native-check.log" 2>&1 ) || true
# One test is EXPECTED to fail in this configuration: t-lock drives libgcrypt's locks from many
# threads, and libgpg-error is built --disable-threads (build-libgpg-error.sh), so those locks are
# no-ops and it aborts on "Assertion `pool_is_locked' failed". A domain runs one thread. Any other
# failure, or t-lock passing (the no-threads build would then not be what it claims), fails the gate.
npass=$(grep -c '^PASS:' "$LOG/native-check.log" || true)
fails=$(grep -E '^(FAIL|ERROR):' "$LOG/native-check.log" | awk '{print $2}' | sort | tr '\n' ' ')
echo "libgcrypt: native check: PASS $npass, SKIP $(grep -c '^SKIP:' "$LOG/native-check.log" || true), FAIL/ERROR: ${fails:-none}"
[ "$fails" = "t-lock " ] && [ "$npass" -ge 30 ] || { echo "libgcrypt: NATIVE CHECK FAILED" >&2; exit 1; }
echo "libgcrypt: native check OK (t-lock fails as expected without threads)"

( cd "$X" && PATH="$TS_DEPS_PREFIX/bin:$PATH" ./configure --host=riscv64-unknown-linux-musl --prefix="$TS_DEPS_PREFIX" "${OPTS[@]}" \
    --with-libgpg-error-prefix="$TS_DEPS_PREFIX" CPPFLAGS="-DCAPSTONE_SINGLE_THREAD_DOMAIN" \
    > "$LOG/cap-configure.log" 2>&1 )
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" make -j16 SUBDIRS="compat mpi cipher random src" > "$LOG/cap-build.log" 2>&1 )
( cd "$X" && make -C src install > "$LOG/cap-install.log" 2>&1 )
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "libgcrypt: libgcrypt.a installed ($(stat -c %s "$TS_DEPS_PREFIX/lib/libgcrypt.a") bytes); cast sites: $(wc -l < "$LOG/cast-sites.txt")"
for t in dumpsexp hmac256 mpicalc; do
  "$CAPSTONE_LLVM_BIN/llvm-readelf" -h "$X/src/$t" > /dev/null || { echo "libgcrypt: $t did not link" >&2; exit 1; }
done
echo "libgcrypt: dumpsexp, hmac256 and mpicalc link as domains"
