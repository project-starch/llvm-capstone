#!/usr/bin/env bash
# The pgbench workload of the established-workload reuse run, made on the build host with
# PostgreSQL 17.5's own pgbench (docs/plans/2026-11-15-postgres-established-workload.md).
#
#   make-pgbench-workload.sh NATIVE_INSTALL FIXTURE_CLUSTER SCALE TRANSACTIONS SEED NEW_OUT
#
# NATIVE_INSTALL is the 16-byte MAXALIGN native build of 17.5 that made FIXTURE_CLUSTER
# (ports/postgres/app/build-native16-fixture.py), so its server writes the on-disk format the
# guest's postgres.dom reads. A protected program cannot fork, so the guest has no postmaster and
# pgbench cannot connect to it. The host therefore runs pgbench against the native server, and the
# guest replays what pgbench sent in one single-user session:
#   1. initialised cluster: a copy of FIXTURE_CLUSTER, a native server on a private Unix socket,
#      `pgbench -i -s SCALE` (default init steps dtgvp: drop, tables, client-side COPY, vacuum,
#      primary keys), clean shutdown. NEW_OUT/cluster is the guest's starting state.
#   2. statements: a fresh copy of NEW_OUT/cluster, the native server again, and
#      `pgbench -n -c 1 -j 1 -t TRANSACTIONS --random-seed=SEED -M simple --debug` (built-in tpcb-like
#      script, no pre-run vacuum). Its debug log prints every statement as sent after variable
#      substitution ("client 0 sending ..."); those lines, in order, are NEW_OUT/tpcb.sql.
#      Step 2 runs twice on separate copies and both statement lists must be identical.
#   3. reference output: the native backend in single-user mode on a fresh copy of NEW_OUT/cluster
#      with the guest's options, reading tpcb.sql; NEW_OUT/native.out. The result rows of every
#      guest run must equal its result rows.
# NEW_OUT/manifest.txt records versions, commands and hashes.
set -euo pipefail
[[ $# == 6 ]] || { echo "usage: $0 NATIVE_INSTALL FIXTURE_CLUSTER SCALE TRANSACTIONS SEED NEW_OUT" >&2; exit 2; }
N=$(realpath "$1") FIX=$(realpath "$2") SCALE=$3 T=$4 SEED=$5 OUT=$(realpath -m "$6")
[[ ! -e $OUT ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
for v in "$SCALE" "$T" "$SEED"; do [[ $v =~ ^[0-9]+$ ]] || { echo "not a number: $v" >&2; exit 2; }; done
[[ $("$N/bin/postgres" --version) == 'postgres (PostgreSQL) 17.5' ]] || { echo "not postgres 17.5" >&2; exit 2; }
[[ $("$N/bin/pgbench" --version) == 'pgbench (PostgreSQL) 17.5' ]] || { echo "not pgbench 17.5" >&2; exit 2; }
mkdir -p "$OUT/sock"
PORT=54329
GUEST_OPTS=(-c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT
            -c dynamic_shared_memory_type=mmap -c exit_on_error=true)
log() { echo "[pgbench-workload] $*"; }

tree_sha() {  # path, then content hash, of every file below $1
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) | sha256sum | cut -d' ' -f1
}

server() {  # start|stop DATA LOG
  if [[ $1 == start ]]; then
    "$N/bin/pg_ctl" -D "$2" -l "$3" -w -o "-k $OUT/sock -p $PORT -c listen_addresses= -c dynamic_shared_memory_type=mmap" start > /dev/null
  else
    "$N/bin/pg_ctl" -D "$2" -m fast -w stop > /dev/null
  fi
}

statements() {  # WORK_DIR -> WORK_DIR/tpcb.sql
  local w=$1
  cp -a "$OUT/cluster" "$w/cluster"
  server start "$w/cluster" "$w/server.log"
  "$N/bin/pgbench" -h "$OUT/sock" -p "$PORT" -n -c 1 -j 1 -t "$T" --random-seed="$SEED" -M simple --debug postgres \
    > "$w/pgbench.out" 2> "$w/pgbench-debug.log"
  server stop "$w/cluster"
  sed -n 's/^pgbench: client 0 sending //p' "$w/pgbench-debug.log" > "$w/tpcb.sql"
  local n
  n=$(wc -l < "$w/tpcb.sql")
  [[ $n == $((7 * T)) ]] || { echo "$w: $n statements, expected $((7 * T))" >&2; exit 2; }
  grep -q "^number of transactions actually processed: $T/$T\$" "$w/pgbench.out" \
    || { echo "$w: pgbench did not process $T transactions" >&2; exit 2; }
}

# 1. the initialised cluster
mkdir -p "$OUT/init"
cp -a "$FIX" "$OUT/init/cluster"
server start "$OUT/init/cluster" "$OUT/init/server.log"
"$N/bin/pgbench" -h "$OUT/sock" -p "$PORT" -i -s "$SCALE" postgres > "$OUT/init/pgbench-init.log" 2>&1
server stop "$OUT/init/cluster"
cp -a "$OUT/init/cluster" "$OUT/cluster"
rm -f "$OUT/cluster/postmaster.opts"
log "initialised cluster at scale $SCALE"

# 2. the statements, twice
mkdir -p "$OUT/gen-1" "$OUT/gen-2"
statements "$OUT/gen-1"
statements "$OUT/gen-2"
cmp "$OUT/gen-1/tpcb.sql" "$OUT/gen-2/tpcb.sql" || { echo "two generations differ" >&2; exit 2; }
cp "$OUT/gen-1/tpcb.sql" "$OUT/tpcb.sql"
log "$T transactions, $(wc -l < "$OUT/tpcb.sql") statements, identical in two generations"

# 3. the reference output
mkdir -p "$OUT/native"
cp -a "$OUT/cluster" "$OUT/native/cluster"
"$N/bin/postgres" --single -D "$OUT/native/cluster" "${GUEST_OPTS[@]}" postgres \
  < "$OUT/tpcb.sql" > "$OUT/native.out" 2>&1
rows=$(grep -c $'^\t 1: abalance = ' "$OUT/native.out" || true)
[[ $rows == "$T" ]] || { echo "native replay printed $rows abalance rows, expected $T" >&2; exit 2; }
log "native single-user replay: $rows abalance rows"

{
  echo "made $(date -u +%FT%TZ) by make-pgbench-workload.sh $(sha256sum "$0" | cut -d' ' -f1)"
  echo "postgres: $("$N/bin/postgres" --version) $(sha256sum "$N/bin/postgres" | cut -d' ' -f1)"
  echo "pgbench: $("$N/bin/pgbench" --version) $(sha256sum "$N/bin/pgbench" | cut -d' ' -f1)"
  echo "fixture: $FIX tree $(tree_sha "$FIX")"
  echo "init: pgbench -i -s $SCALE postgres (default init steps dtgvp)"
  echo "run: pgbench -n -c 1 -j 1 -t $T --random-seed=$SEED -M simple --debug postgres (built-in tpcb-like)"
  echo "native replay: postgres --single -D CLUSTER ${GUEST_OPTS[*]} postgres < tpcb.sql"
  echo "cluster tree: $(tree_sha "$OUT/cluster")"
  sha256sum "$OUT/tpcb.sql" "$OUT/native.out" | sed "s|$OUT/||"
  echo "pgbench report (generation 1):"
  sed 's/^/  /' "$OUT/gen-1/pgbench.out"
} > "$OUT/manifest.txt"
rm -rf "$OUT/init/cluster" "$OUT/gen-1/cluster" "$OUT/gen-2/cluster" "$OUT/native/cluster" "$OUT/sock"
cat "$OUT/manifest.txt"
