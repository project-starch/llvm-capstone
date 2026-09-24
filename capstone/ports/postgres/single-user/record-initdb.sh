#!/usr/bin/env bash
# Record what initdb asks the backend to do: every `postgres` it runs, with its
# arguments, its environment, its standard input, and the data directory as it
# stood before it. initdb is a separate program the port does not build -- the
# domain image is the backend alone -- so run-domain.sh replays these
# invocations one at a time from the recorded state: the catalog script under
# --boot first, then each setup script under --single.
#
#   bash record-initdb.sh
#
# Output, under $PG_SU_ROOT/initdb-rec/: call-<n>.args (one argument per line),
# call-<n>.env (the PG* variables initdb set), call-<n>.stdin for the calls that
# read one, pgdata-before-<n>/ for those same calls, and initdb.log. Uses the
# native build the survey made ($PG_SU_ROOT/pg-native), and builds it when it is
# gone, with the survey's configure line.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}

if [[ ! -x "$ROOT/pg-native/bin/postgres" ]]; then
  [[ -f "$TARBALL" ]] || { echo "no tarball $TARBALL" >&2; exit 2; }
  mkdir -p "$ROOT/native"
  cd "$ROOT/native"
  [[ -d postgresql-17.5 ]] || tar xjf "$TARBALL"
  cd postgresql-17.5
  ./configure --prefix="$ROOT/pg-native" --without-readline --without-zlib --without-icu > "$ROOT/native-configure.log" 2>&1
  make -j"$(nproc)" > "$ROOT/native-make.log" 2>&1
  make install > "$ROOT/native-install.log" 2>&1
  echo "[record-initdb] built $ROOT/pg-native"
fi

# initdb finds the backend and share/ next to its own binary, so the recorder
# is a copy of the install whose bin/postgres is a script around the real one.
REC=$ROOT/initdb-rec
PFX=$ROOT/rec-prefix
rm -rf "$REC" "$PFX"
mkdir -p "$REC"
cp -a "$ROOT/pg-native" "$PFX"
mv "$PFX/bin/postgres" "$PFX/bin/postgres.real"
cat > "$PFX/bin/postgres" <<'EOF'
#!/usr/bin/env bash
# initdb's backend, recorded; written by record-initdb.sh. Every call is logged;
# the ones that read their input (--boot, --single) also get the input and the
# data directory they started from.
REC=${PGSU_REC:?}
n=$(( $(cat "$REC/count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$REC/count"
printf '%s\n' "$@" > "$REC/call-$n.args"
{ env | grep -E '^(PG|LC_|LANG)' | sort; } > "$REC/call-$n.env" || true
takes_input=0
for a in "$@"; do [[ $a == --boot || $a == --single ]] && takes_input=1; done
if (( takes_input )); then
  [[ -n ${PGDATA:-} && -d $PGDATA ]] && cp -a "$PGDATA" "$REC/pgdata-before-$n"
  tee "$REC/call-$n.stdin" | exec "$0.real" "$@"
else
  exec "$0.real" "$@"
fi
EOF
chmod 0755 "$PFX/bin/postgres"

# C locale throughout and UTF8, so the catalog carries no host locale; the
# superuser is "pg", as in the survey.
D=$REC/pgdata
PGSU_REC=$REC "$PFX/bin/initdb" -D "$D" -U pg --no-sync --no-locale -E UTF8 > "$REC/initdb.log" 2>&1 \
  || { tail -5 "$REC/initdb.log" >&2; echo "initdb failed (see $REC/initdb.log)" >&2; exit 1; }
n=$(cat "$REC/count")
echo "[record-initdb] $n backend calls recorded under $REC"
for ((i = 1; i <= n; i++)); do
  printf '  call %2d: %s' "$i" "$(tr '\n' ' ' < "$REC/call-$i.args")"
  [[ -f "$REC/call-$i.stdin" ]] && printf ' (input %s bytes, pgdata-before-%d)' "$(wc -c < "$REC/call-$i.stdin")" "$i"
  echo
done
