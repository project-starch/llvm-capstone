#!/usr/bin/env bash
# The guest-side programs of the established memcached workload (pre-registration
# docs/plans/2026-11-15-memcached-established-workload.md), built natively and statically for riscv64 with the
# buildroot cross toolchain. They run unprotected in the guest beside the protected memcached.dom:
#   mc-crusher   the load generator of memcached's maintainer, github.com/memcached/mc-crusher at MCC_COMMIT,
#                compiled with the line of its own Makefile plus -static, against
#   libevent     2.1.12-stable, the tarball and sha256 that ports/memcached/app/deps/deps.json pins;
#   mc-harness   the port's oracle driver, ports/memcached/app/host/mc-harness/mc-harness.c, whose --load
#                phase starts mc-crusher after the oracle and counts the commands the server processed.
#   build-load-client.sh NEW_OUTPUT  ->  NEW_OUTPUT/{mc-crusher,mc-harness,SOURCES.txt}
# With MCC_PATCH set to a patch file, mc-crusher is patched after the checkout (patch -p1, fuzz 0), and
# SOURCES.txt records the patch and its sha256: mc-crusher-fixed-seed.patch fixes the key seed
# (docs/plans/2026-11-15-memcached-fixed-seed.md).
# Downloads go to NEW_OUTPUT/dl; nothing downloaded is executed except libevent's configure and make.
set -euo pipefail
HOOKS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
APP=$(cd -- "$HOOKS/../../../../ports/memcached/app" && pwd)
[[ $# == 1 ]] || { echo 'usage: build-load-client.sh NEW_OUTPUT' >&2; exit 2; }
OUT=$(realpath -m "$1")
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
BR=${CAPSTONE_BUILDROOT_DIR:-$HOME/caplifive-system/sw/buildroot}
CROSS=$BR/build/host/bin/riscv64-buildroot-linux-gnu-
MCC_REPO=https://github.com/memcached/mc-crusher.git
MCC_COMMIT=7ca3f14c128495a509cd0aaff464545377fa5cee
[[ -x ${CROSS}gcc ]] || { echo "no cross compiler at ${CROSS}gcc" >&2; exit 2; }
mkdir -p "$OUT/dl" "$OUT/build" "$OUT/logs"

read -r LE_URL LE_SHA < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["libevent"]; print(d["url"], d["sha256"])' \
                          "$APP/deps/deps.json")
LE_TAR=$OUT/dl/$(basename "$LE_URL")
curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "$LE_TAR" "$LE_URL"
echo "$LE_SHA  $LE_TAR" | sha256sum -c --quiet - || { echo "libevent: sha256 MISMATCH" >&2; exit 1; }

git clone -q "$MCC_REPO" "$OUT/dl/mc-crusher"
git -C "$OUT/dl/mc-crusher" -c advice.detachedHead=false checkout -q --detach "$MCC_COMMIT"
[[ $(git -C "$OUT/dl/mc-crusher" rev-parse HEAD) == "$MCC_COMMIT" ]] || { echo "mc-crusher: not at $MCC_COMMIT" >&2; exit 1; }
if [[ -n ${MCC_PATCH:-} ]]; then
  MCC_PATCH=$(realpath "$MCC_PATCH")
  ( cd "$OUT/dl/mc-crusher" && patch --batch --forward --fuzz=0 -p1 < "$MCC_PATCH" ) > "$OUT/logs/mc-crusher-patch.log" 2>&1 ||
    { echo "mc-crusher: $MCC_PATCH does not apply"; cat "$OUT/logs/mc-crusher-patch.log"; exit 1; }
fi

LE=$OUT/build/libevent PREFIX=$OUT/build/prefix
mkdir -p "$LE"
tar -xf "$LE_TAR" -C "$LE" --strip-components=1
( cd "$LE" && ./configure --host=riscv64-buildroot-linux-gnu CC="${CROSS}gcc" AR="${CROSS}ar" RANLIB="${CROSS}ranlib" \
    --prefix="$PREFIX" --disable-shared --enable-static --disable-openssl --disable-samples \
    --disable-libevent-regress --disable-debug-mode > "$OUT/logs/libevent-configure.log" 2>&1 &&
  make -j"${JOBS:-8}" > "$OUT/logs/libevent-build.log" 2>&1 &&
  make install > "$OUT/logs/libevent-install.log" 2>&1 ) || { echo "libevent: build FAILED (see $OUT/logs)" >&2; exit 1; }

M=$OUT/dl/mc-crusher
"${CROSS}gcc" -g -O2 -pthread -static -I"$PREFIX/include" -o "$OUT/mc-crusher" \
  "$M/mc-crusher.c" "$M/pcg-basic.c" "$M/itoa_ljust.c" -L"$PREFIX/lib" -levent -lm > "$OUT/logs/mc-crusher.log" 2>&1 ||
  { echo "mc-crusher: build FAILED"; cat "$OUT/logs/mc-crusher.log"; exit 1; }
"${CROSS}gcc" -O1 -Wall -static -o "$OUT/mc-harness" "$APP/host/mc-harness/mc-harness.c" -lpthread \
  > "$OUT/logs/mc-harness.log" 2>&1 || { echo "mc-harness: build FAILED"; cat "$OUT/logs/mc-harness.log"; exit 1; }

for f in mc-crusher mc-harness; do
  file "$OUT/$f" | grep -q 'statically linked' || { echo "$f: not statically linked" >&2; exit 1; }
done
{
  echo "libevent $LE_URL sha256 $LE_SHA"
  echo "mc-crusher $MCC_REPO commit $MCC_COMMIT tree $(git -C "$M" rev-parse 'HEAD^{tree}')"
  [[ -z ${MCC_PATCH:-} ]] || echo "mc-crusher patch $(basename "$MCC_PATCH") sha256 $(sha256sum < "$MCC_PATCH" | cut -d' ' -f1)"
  echo "mc-harness.c sha256 $(sha256sum < "$APP/host/mc-harness/mc-harness.c" | cut -d' ' -f1)"
  echo "cross compiler $("${CROSS}gcc" --version | head -1)"
  for f in mc-crusher mc-harness; do echo "$f sha256 $(sha256sum < "$OUT/$f" | cut -d' ' -f1)"; done
} | tee "$OUT/SOURCES.txt"
