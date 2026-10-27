#!/bin/bash
# Slot census of Redis and memcached under YCSB (Cooper et al., SoCC 2010), the named key-value workload whose size
# knob is the data set. One cell per server and record count: start the server under the tracer, YCSB load phase,
# then core workloads A (50% read, 50% update) and C (100% read) with $OPS operations each, all with YCSB's default
# record (10 fields of 100 bytes), 64 client threads (as AtomicDisk, FAST 2025), uniform request distribution. Then
# stop the server so the tracer reports.
#
# usage: run-ycsb.sh OUT_DIR [RECORDS...]   (default 100000 1000000 5000000 16000000)
# 5,000,000 records with 1 M operations is AtomicDisk's Redis configuration (FAST 2025, Fig. 5); 16,000,000 records of
# about 1 KB is Lersch et al.'s Redis/memcached configuration (PVLDB 2020, sec. 5.4). Redis is built with MALLOC=libc
# so every object goes through malloc; memcached keeps items in its slab allocator (hidden from the tracer, its
# curr_items reported beside the census). Downloads YCSB $YCSB (full release: it carries the redis and memcached
# bindings), a Temurin JDK $JDK (YCSB 0.17 needs Java 8 or 11; the hosts have none), Redis and memcached, into
# $SLOTCENSUS_CACHE (default ~/slot-census/cache), and records their sha256. A cell is valid when every YCSB phase
# exits 0 with every operation returning OK, the server holds exactly the loaded records afterwards (Redis: DBSIZE =
# records + 1, its "_indices" sorted set; memcached: curr_items = records), and the server exits 0.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
RECORDS=("$@"); [ ${#RECORDS[@]} -gt 0 ] || RECORDS=(100000 1000000 5000000 16000000)
YCSB=0.17.0
JDK=11.0.25_9
REDIS=7.2.16
MEMCACHED=1.6.45
OPS=${SLOTCENSUS_YCSB_OPS:-1000000}
THREADS=64
PORT_R=${SLOTCENSUS_REDIS_PORT:-16381}
PORT_M=${SLOTCENSUS_MEMCACHED_PORT:-21213}
. "$HERE/tracer.sh"
. "$HERE/libevent.sh"
start_pass

Y_T=ycsb-$YCSB.tar.gz J_T=OpenJDK11U-jdk_x64_linux_hotspot_$JDK.tar.gz R_T=redis-$REDIS.tar.gz
M_T=memcached-$MEMCACHED.tar.gz
cached "$Y_T" "https://github.com/brianfrankcooper/YCSB/releases/download/$YCSB/$Y_T"
cached "$J_T" "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-${JDK/_/%2B}/$J_T"
cached "$R_T" "https://download.redis.io/releases/$R_T"
cached "$M_T" "https://memcached.org/files/$M_T"
tar xzf "$CACHE/$Y_T" -C "$TOOL" ycsb-$YCSB/lib ycsb-$YCSB/workloads ycsb-$YCSB/redis-binding \
  ycsb-$YCSB/memcached-binding || exit 1
mkdir -p "$TOOL/jdk" && tar xzf "$CACHE/$J_T" -C "$TOOL/jdk" --strip-components=1 || exit 1
tar xzf "$CACHE/$R_T" -C "$TOOL" && tar xzf "$CACHE/$M_T" -C "$TOOL" || exit 1
Y=$TOOL/ycsb-$YCSB JAVA=$TOOL/jdk/bin/java
R=$TOOL/redis-$REDIS M=$TOOL/memcached-$MEMCACHED
make -C "$R" -j16 MALLOC=libc BUILD_TLS=no > "$OUT/logs/redis-build.log" 2>&1 \
  && "$R/src/redis-server" --version | grep -q "malloc=libc" || { echo "run-ycsb: redis build failed" >&2; exit 1; }
build_libevent
(cd "$M" && ./configure --with-libevent="$TOOL/libevent" > "$OUT/logs/memcached-build.log" 2>&1 &&
 make -j16 >> "$OUT/logs/memcached-build.log" 2>&1) || { echo "run-ycsb: memcached build failed" >&2; exit 1; }
"$JAVA" -version > "$OUT/logs/java.txt" 2>&1 || { echo "run-ycsb: java does not run" >&2; exit 1; }
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "$Y_T" "$J_T" "$R_T" "$M_T")
  echo "java $(head -1 "$OUT/logs/java.txt")"
  echo "redis $("$R/src/redis-server" --version) (make MALLOC=libc BUILD_TLS=no)"
  echo "memcached $("$M/memcached" -V) (libevent $LIBEVENT static, sha256 $LIBEVENT_SHA)"
  echo "ycsb $YCSB: load, then workloads A and C with operationcount=$OPS, $THREADS threads, default 10x100-byte fields"
  echo "records ${RECORDS[*]}"
} > "$OUT/provenance.txt"

# phase: -load or -t (the transaction phase). YCSB exits 0 even when it rejects its arguments (it prints its usage),
# so a phase's exit status proves nothing: ok_count is the check
ycsb() {   # phase(-load|-t) binding class workload records log [props...]
  local phase=$1 binding=$2 class=$3 workload=$4 records=$5 log=$6; shift 6
  "$JAVA" -cp "$Y/lib/*:$Y/$binding/lib/*" site.ycsb.Client "$phase" -db "$class" -P "$Y/workloads/$workload" \
    -p recordcount="$records" -p operationcount="$OPS" -threads "$THREADS" -s "$@" > "$log" 2>&1
}
# every operation of the phase returned OK: the sum of "[OP], Return=OK, n" equals the expected count, with no
# other return code
ok_count() {   # log expected
  python3 - "$1" "$2" <<'EOF'
import re, sys
text = open(sys.argv[1], errors="replace").read()
ok = sum(int(n) for n in re.findall(r"^\[(?:INSERT|READ|UPDATE)\], Return=OK, (\d+)", text, re.M))
bad = re.findall(r"^\[(?:INSERT|READ|UPDATE)\], Return=(?!OK)\w+, (\d+)", text, re.M)
sys.exit(0 if ok == int(sys.argv[2]) and not bad else 1)
EOF
}

up() { (exec 3<> "/dev/tcp/127.0.0.1/$1") 2> /dev/null; }
mc_say() {   # port command
  exec 3<> "/dev/tcp/127.0.0.1/$1" || return 1
  printf '%s\r\n' "$2" >&3
  timeout 5 sed -u '/^END\r*$/q' <&3
  exec 3<&-
}

# use_server SERVER: its port, measured executable, YCSB binding, client class and connection properties
use_server() {
  if [ "$1" = redis ]; then
    port=$PORT_R exe=redis-server binding=redis-binding class=site.ycsb.db.RedisClient
    props=(-p redis.host=127.0.0.1 -p "redis.port=$PORT_R")
  else
    port=$PORT_M exe=memcached binding=memcached-binding class=site.ycsb.db.MemcachedClient
    props=(-p "memcached.hosts=127.0.0.1:$PORT_M")
  fi
}

# start_server SERVER CELL: the measured server in the background (srv its pid), once its port answers
start_server() {
  up $port && { echo "run-ycsb: port $port is taken" >&2; exit 1; }
  if [ "$1" = redis ]; then
    measured "$OUT/$2.jsonl" \
      "$R/src/redis-server" --port $port --bind 127.0.0.1 --save "" --appendonly no --set-proc-title no \
      --dir "$OUT" > "$OUT/logs/$2.server" 2>&1 &
  else
    measured "$OUT/$2.jsonl" \
      "$M/memcached" -l 127.0.0.1 -p $port -U 0 -m 65536 -t 4 -c 4096 -A > "$OUT/logs/$2.server" 2>&1 &
  fi
  srv=$!
  for _ in $(seq 1 100); do up $port && break; sleep 0.1; done
}

# stop_server SERVER RECORDS: the items it holds (n) against what it must hold (want), then a clean shutdown
stop_server() {
  if [ "$1" = redis ]; then
    n=$("$R/src/redis-cli" -p $port dbsize); want=$(($2 + 1))
    "$R/src/redis-cli" -p $port shutdown nosave > /dev/null 2>&1
  else
    n=$(mc_say $port stats | tr -d '\r' | sed -n 's/^STAT curr_items //p'); want=$2
    mc_say $port shutdown > /dev/null 2>&1
  fi
}

: > "$OUT/cells.tsv"
for records in "${RECORDS[@]}"; do
  for server in redis memcached; do
    cell=ycsb-$server-$records
    t=$(date +%s)
    use_server $server
    start_server $server "$cell"
    ycsb -load "$binding" "$class" workloada "$records" "$OUT/logs/$cell.load" "${props[@]}"; load=$?
    ycsb -t "$binding" "$class" workloada "$records" "$OUT/logs/$cell.a" "${props[@]}"; a=$?
    ycsb -t "$binding" "$class" workloadc "$records" "$OUT/logs/$cell.c" "${props[@]}"; c=$?
    ok_count "$OUT/logs/$cell.load" "$records"; okl=$?
    ok_count "$OUT/logs/$cell.a" "$OPS"; oka=$?
    ok_count "$OUT/logs/$cell.c" "$OPS"; okc=$?
    stop_server $server "$records"
    wait $srv; rc=$?
    {
      for ph in load a c; do
        echo "== $ph"
        grep -E '^\[(OVERALL|INSERT|READ|UPDATE)\], (RunTime|Throughput|Operations|Return)' "$OUT/logs/$cell.$ph"
      done
      echo "items $n (want $want)"
    } > "$OUT/$cell.out"
    v=INVALID
    [ "$rc$load$a$c$okl$oka$okc" = 0000000 ] && [ "${n:-x}" = "$want" ] && v=valid
    record_cell "$cell" "$exe" "$v" "rc=$rc phases=$load$a$c ok=$okl$oka$okc items=${n:-?} $(took $t)"
  done
done
finish_pass
