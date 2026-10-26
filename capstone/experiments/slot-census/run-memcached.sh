#!/bin/bash
# Slot census of a memcached server across dataset sizes. memcached keeps its items in its own slab allocator,
# carved from 1 MiB pages it mallocs, so the tracer sees pages, not items: each cell records both the census and the
# server's own item count, which is the measure of what a nested allocator hides.
#
# usage: run-memcached.sh OUT_DIR [KEYS...]   (default 10000 100000 1000000 10000000)
# Downloads memcached-$VERSION from memcached.org and libevent-$LIBEVENT (built static, as the measuring host has no
# libevent headers) from its GitHub release into $SLOTCENSUS_CACHE (default ~/slot-census/cache) and records their
# sha256. One cell per key count: start the server under the tracer (-m 8192, so nothing is evicted), run
# mc-load.py (set every key with a 100-byte value, get each once, delete a tenth), then send "shutdown" (-A), and the
# server returns from main, so the tracer reports. A cell is valid when the load's replies were all as expected and
# the server exited 0.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
KEYS=("$@"); [ ${#KEYS[@]} -gt 0 ] || KEYS=(10000 100000 1000000 10000000)
VERSION=1.6.45
LIBEVENT=2.1.12-stable
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}
PORT=${SLOTCENSUS_MEMCACHED_PORT:-21211}
[ -e "$OUT" ] && { echo "run-memcached: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT" "$CACHE"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

T=memcached-$VERSION.tar.gz E=libevent-$LIBEVENT.tar.gz
[ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" "https://memcached.org/files/$T" || exit 1
[ -s "$CACHE/$E" ] || curl -fsSL -o "$CACHE/$E" \
  "https://github.com/libevent/libevent/releases/download/release-$LIBEVENT/$E" || exit 1
tar xzf "$CACHE/$T" -C "$TOOL" && tar xzf "$CACHE/$E" -C "$TOOL" || exit 1
M=$TOOL/memcached-$VERSION
(cd "$TOOL/libevent-$LIBEVENT" && ./configure --prefix="$TOOL/libevent" --disable-shared --disable-openssl \
   > "$OUT/build.log" 2>&1 && make -j16 install >> "$OUT/build.log" 2>&1 &&
 cd "$M" && ./configure --with-libevent="$TOOL/libevent" >> "$OUT/build.log" 2>&1 &&
 make -j16 >> "$OUT/build.log" 2>&1) \
  || { echo "run-memcached: build failed" >&2; exit 1; }
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "$T" "$E")
  echo "memcached $("$M/memcached" -V)"
  echo "build libevent $LIBEVENT static (--disable-openssl); memcached ./configure --with-libevent && make"
  echo "keys ${KEYS[*]} value 100 bytes; server -m 8192 -t 4 -A"
} > "$OUT/provenance.txt"

: > "$OUT/cells.tsv"
for keys in "${KEYS[@]}"; do
  cell=memcached-$keys
  (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null && { echo "run-memcached: port $PORT is taken" >&2; exit 1; }
  t=$(date +%s)
  SLOTCENSUS_OUT=$OUT/$cell.jsonl LD_PRELOAD=$TOOL/slotcensus.so \
    "$M/memcached" -l 127.0.0.1 -p "$PORT" -U 0 -m 8192 -t 4 -A > "$OUT/$cell.server" 2>&1 &
  srv=$!
  for i in $(seq 1 100); do (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null && break; sleep 0.1; done
  python3 "$HERE/mc-load.py" "$PORT" "$keys" 100 > "$OUT/$cell.out" 2>&1
  load=$?
  printf 'shutdown\r\n' | python3 -c "import socket,sys; s=socket.create_connection(('127.0.0.1', $PORT)); \
s.sendall(sys.stdin.buffer.read()); s.recv(64)" 2> /dev/null
  wait $srv; rc=$?
  v=INVALID; [ "$rc" = 0 ] && [ "$load" = 0 ] && v=valid
  printf '%s\tmemcached\t%s\trc=%s load=%s %ss\n' "$cell" "$v" "$rc" "$load" "$(( $(date +%s) - t ))" \
    | tee -a "$OUT/cells.tsv"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
