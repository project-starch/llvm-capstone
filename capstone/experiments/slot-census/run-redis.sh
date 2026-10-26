#!/bin/bash
# Slot census of a Redis server across dataset sizes. Redis is built with MALLOC=libc so that every object goes
# through malloc (its default build uses jemalloc, which the tracer would not see). One cell per key count: start
# the server under the tracer, DEBUG POPULATE <keys> keys of 100-byte values, run redis-benchmark's mixed commands
# over the same key space, record DBSIZE and INFO memory, then SHUTDOWN NOSAVE (the server exits through exit(),
# so the tracer reports).
#
# usage: run-redis.sh OUT_DIR [KEYS...]   (default 10000 100000 1000000 10000000)
# Downloads redis-$VERSION from download.redis.io into $SLOTCENSUS_CACHE (default ~/slot-census/cache) and records
# its sha256. A cell is valid when the server answered every step and DBSIZE is at least the populated key count.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
KEYS=("$@"); [ ${#KEYS[@]} -gt 0 ] || KEYS=(10000 100000 1000000 10000000)
VERSION=7.2.16
CACHE=${SLOTCENSUS_CACHE:-$HOME/slot-census/cache}
PORT=${SLOTCENSUS_REDIS_PORT:-16379}
REQUESTS=${SLOTCENSUS_REDIS_REQUESTS:-200000}
[ -e "$OUT" ] && { echo "run-redis: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT" "$CACHE"; OUT=$(cd "$OUT" && pwd); TOOL=$OUT/tool
. "$HERE/tracer.sh"
build_tracer
run_controls

T=redis-$VERSION.tar.gz
[ -s "$CACHE/$T" ] || curl -fsSL -o "$CACHE/$T" "https://download.redis.io/releases/$T" || exit 1
tar xzf "$CACHE/$T" -C "$TOOL" || exit 1
R=$TOOL/redis-$VERSION
make -C "$R" -j16 MALLOC=libc BUILD_TLS=no > "$OUT/build.log" 2>&1 || { echo "run-redis: build failed" >&2; exit 1; }
"$R/src/redis-server" --version | grep -q "malloc=libc" || { echo "run-redis: not a libc build" >&2; exit 1; }
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "$T")
  echo "redis $("$R/src/redis-server" --version)"
  echo "build make MALLOC=libc BUILD_TLS=no"
  echo "keys ${KEYS[*]} value 100 bytes; redis-benchmark -n $REQUESTS -r <keys>, default command set"
} > "$OUT/provenance.txt"

cli() { "$R/src/redis-cli" -p "$PORT" "$@"; }
: > "$OUT/cells.tsv"
for keys in "${KEYS[@]}"; do
  cell=redis-$keys
  cli ping > /dev/null 2>&1 && { echo "run-redis: port $PORT is taken" >&2; exit 1; }
  t=$(date +%s)
  SLOTCENSUS_OUT=$OUT/$cell.jsonl LD_PRELOAD=$TOOL/slotcensus.so \
    "$R/src/redis-server" --port "$PORT" --bind 127.0.0.1 --save "" --appendonly no --set-proc-title no \
    --enable-debug-command local --dir "$OUT" > "$OUT/$cell.server" 2>&1 &
  srv=$!
  for i in $(seq 1 100); do cli ping > /dev/null 2>&1 && break; sleep 0.1; done
  {
    echo "populate $(cli debug populate "$keys" key 100)"
    "$R/src/redis-benchmark" -p "$PORT" -n "$REQUESTS" -r "$keys" -d 100 -q
    echo "dbsize $(cli dbsize)"
    cli info memory | grep -E '^used_memory:|^used_memory_peak:'
  } > "$OUT/$cell.out" 2>&1
  cli shutdown nosave > /dev/null 2>&1
  wait $srv; rc=$?
  n=$(sed -n 's/^dbsize //p' "$OUT/$cell.out")
  v=INVALID; [ "$rc" = 0 ] && grep -q '^populate OK' "$OUT/$cell.out" && [ "${n:-0}" -ge "$keys" ] && v=valid
  printf '%s\tredis-server\t%s\trc=%s dbsize=%s %ss\n' "$cell" "$v" "$rc" "${n:-?}" "$(( $(date +%s) - t ))" \
    | tee -a "$OUT/cells.tsv"
done
echo CENSUS_DONE >> "$OUT/cells.tsv"
