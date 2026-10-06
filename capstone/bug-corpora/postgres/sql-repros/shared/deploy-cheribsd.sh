#!/bin/bash
# Deploy the purecap PostgreSQL tree into the CheriBSD guest.
#
#   deploy-cheribsd.sh [stage prefix]
#
# Separate from the runners on purpose: it is slow, it changes what every
# later run measures, and doing it implicitly is how a run ends up naming a
# build it did not use. The runners check what the guest has and refuse or
# mark cases not-applicable; this is what changes the answer.
#
# It also re-initdbs, because the catalog has to come from the same binaries:
# a cluster from an older tree can be silently incompatible with a newer one.
set -uo pipefail
STAGE=${1:-$HOME/arms/postgres/shared/stage-purecap/usr/local/pgsql}
PORT=${PG_CHERI_PORT:-10086}
BASE=${PG_CHERI_BASE:-/home/pg/data}
K="-i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no"
K="$K -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
G() { ssh -n $K -p "$PORT" root@localhost "$@" 2>&1; }

[ -x "$STAGE/bin/postgres" ] || { echo "no postgres under $STAGE" >&2; exit 2; }
echo "staging from $STAGE"
ls "$STAGE/lib"/*.so 2>/dev/null | sed 's#.*/#  lib #'
ls "$STAGE/share/extension"/*.control 2>/dev/null | sed 's#.*/#  ext #'

TAR=/tmp/pgsql-purecap-$(date -u +%Y%m%d-%H%M%S).tgz
( cd "$(dirname "$(dirname "$STAGE")")" && tar czf "$TAR" "usr/local/$(basename "$STAGE")" ) \
  || { echo "tar failed" >&2; exit 2; }
echo "tarball $(du -h "$TAR" | cut -f1)"

scp $K -P "$PORT" "$TAR" root@localhost:/tmp/pgsql-purecap.tgz >/dev/null \
  || { echo "scp failed" >&2; exit 2; }
echo "unpacking"
G 'rm -rf /usr/local/pgsql && cd / && tar xzf /tmp/pgsql-purecap.tgz && /usr/local/pgsql/bin/postgres --version'

echo "extensions now in the guest:"
G 'ls /usr/local/pgsql/share/extension/*.control 2>/dev/null | sed "s#.*/#  #"'

echo "initdb (slow under QEMU)"
G 'pw useradd pg -d /home/pg -m -s /bin/sh 2>/dev/null; chown pg /home/pg'
G "rm -rf $BASE"
G "su -m pg -c '/usr/local/pgsql/bin/initdb -D $BASE -U pg --no-locale -E UTF8'" | tail -4
G "test -f $BASE/PG_VERSION && echo CLUSTER-OK" | tail -1
