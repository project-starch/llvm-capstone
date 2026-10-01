#!/usr/bin/env bash
# M3's instrument: memcached built with -DMC_CAPSTONE_WORKER_MARKER (patch 0004), natively and as a
# domain, so each prints "MC-WORKER <index> conn" when a worker thread sets up a connection. These are
# separate images in $MC_WORK/marker; the oracle's images (build-native.sh, build-domain.sh) stay
# marker-free, and this script checks both directions: the marker images carry the string, the
# oracle images do not.
#   native: patch 0004 only, against build-native.sh's libevent;
#   domain: every patch, the deps SDK (the shrink arm's runtime).
# Output: $MC_WORK/marker/{memcached-native,memcached.dom}.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
[ -f "$MC_WORK/native/prefix/lib/libevent_core.a" ] || { echo "no native libevent: run host/build-native.sh" >&2; exit 2; }
[ -f "$MC_DEPS_PREFIX/lib/libevent_core.a" ] || { echo "no libevent in $MC_DEPS_PREFIX: run deps/build-libevent.sh" >&2; exit 2; }
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
OUT=$MC_WORK/marker; rm -rf "$OUT"; mkdir -p "$OUT/logs"
MARK=$APP/patches/memcached-1.6.45-0004-worker-marker.patch
CFG=(--disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs)

cp -a "$MSRC" "$OUT/native-src"
( cd "$OUT/native-src" && patch --batch --forward --fuzz=0 -p1 < "$MARK" > "$OUT/logs/native-patch.log" &&
  env -u CC -u AR -u RANLIB ./configure --with-libevent="$MC_WORK/native/prefix" "${CFG[@]}" \
    CPPFLAGS=-DMC_CAPSTONE_WORKER_MARKER > "$OUT/logs/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j"${JOBS:-8}" memcached > "$OUT/logs/native-build.log" 2>&1 ) \
  || { echo "native marker build FAILED ($OUT/logs)"; exit 1; }
cp "$OUT/native-src/memcached" "$OUT/memcached-native"

cp -a "$MSRC" "$OUT/domain-src"
for p in "$APP"/patches/*.patch; do
  ( cd "$OUT/domain-src" && patch --batch --forward --fuzz=0 -p1 < "$p" > "$OUT/logs/domain-patch-$(basename "$p").log" ) \
    || { echo "patch $p FAILED" >&2; exit 1; }
done
( cd "$OUT/domain-src" && ./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" "${CFG[@]}" \
    CPPFLAGS=-DMC_CAPSTONE_WORKER_MARKER > "$OUT/logs/domain-configure.log" 2>&1 &&
  make -j"${JOBS:-8}" memcached > "$OUT/logs/domain-build.log" 2>&1 ) || { echo "domain marker build FAILED ($OUT/logs)"; exit 1; }
cp "$OUT/domain-src/memcached" "$OUT/memcached.dom"

# Both directions: the define reached the code (the marker images carry the format string), and it
# did not reach the oracle's images (which must not print it).
has() { python3 -c 'import sys; sys.exit(0 if b"MC-WORKER %d conn" in open(sys.argv[1],"rb").read() else 1)' "$1"; }
for f in "$OUT/memcached-native" "$OUT/memcached.dom"; do
  has "$f" || { echo "MARKER GATE: $f lacks the marker string: the define did not reach thread.c"; exit 1; }
done
for f in "$MC_WORK/native/bin/memcached" "$MC_WORK/domain/memcached-level0.dom" "$MC_WORK/domain/memcached-shrink.dom" \
         "$MC_WORK/domain/memcached-sublet.dom"; do
  [ -f "$f" ] || continue
  ! has "$f" || { echo "MARKER GATE: the oracle image $f carries the marker"; exit 1; }
done
echo "marker: native $(sha256sum < "$OUT/memcached-native" | cut -c1-16) domain $(sha256sum < "$OUT/memcached.dom" | cut -c1-16);" \
  "both carry the marker, the oracle images do not"
