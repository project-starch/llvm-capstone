#!/usr/bin/env bash
# What `postgres --single` asks the operating system for: build PostgreSQL
# natively, initdb, run one single-user session over work.sql, and record the
# syscall census of both under strace. The census is the list of what a domain
# runtime would have to serve; results/<date>/*-syscalls.txt are its output.
#
#   bash survey-native.sh [<postgresql-17.5.tar.bz2>]
#
# Everything goes under $PG_SU_ROOT (default $CAPSTONE_TMP_ROOT/pg-single-user).
# The tarball defaults to the mmgr port's download ($CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}
[[ -f "$TARBALL" ]] || { echo "no tarball $TARBALL" >&2; exit 2; }
command -v strace >/dev/null || { echo "strace needed" >&2; exit 2; }
mkdir -p "$ROOT/native"
cd "$ROOT/native"
[[ -d postgresql-17.5 ]] || tar xjf "$TARBALL"
cd postgresql-17.5
# readline, zlib and icu are left out so that the native and the capstone
# configuration answer the same questions.
./configure --prefix="$ROOT/pg-native" --without-readline --without-zlib --without-icu > "$ROOT/native-configure.log" 2>&1
make -j"$(nproc)" > "$ROOT/native-make.log" 2>&1
make install > "$ROOT/native-install.log" 2>&1
B=$ROOT/pg-native/bin; D=$ROOT/pgdata
rm -rf "$D"
# -c: one summary table per run, no paths in it, which is what gets committed.
strace -f -c -o "$ROOT/initdb-syscalls.txt" "$B/initdb" -D "$D" -U pg --no-sync > "$ROOT/initdb.log" 2>&1
# shared_buffers small, as a domain would run it; dsm through mmap so that the
# census shows the file-backed MAP_SHARED it takes.
strace -f -c -o "$ROOT/single-user-syscalls.txt" "$B/postgres" --single -D "$D" \
  -c shared_buffers=8MB -c dynamic_shared_memory_type=mmap postgres < "$SCRIPT_DIR/work.sql" > "$ROOT/single.log" 2>&1
grep -q 'count = "1500"' "$ROOT/single.log" || { echo "the session did not run work.sql to its end (see $ROOT/single.log)" >&2; exit 1; }
# The full trace, kept here and not committed (it carries the host's paths).
strace -f -o "$ROOT/single.trace" "$B/postgres" --single -D "$D" \
  -c shared_buffers=8MB -c dynamic_shared_memory_type=mmap postgres < "$SCRIPT_DIR/work.sql" > /dev/null 2>&1
echo "processes in the session: $(awk '{print $1}' "$ROOT/single.trace" | sort -u | wc -l)"
echo "census: $ROOT/single-user-syscalls.txt $ROOT/initdb-syscalls.txt"
