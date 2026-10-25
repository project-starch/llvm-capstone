#!/usr/bin/env bash
# PostgreSQL 17.5 backend with the reuse-distance hooks, one arm, virtual profile.
#   usage: build.sh <stock|protected> <new output dir>
# Needs the environment build-virtual.sh needs (CAPSTONE_REPO_ROOT,
# CAPSTONE_LLVM_BUILD_DIR, CAPSTONE_TMP_ROOT holding pg-mmgr-host/pg.tar.bz2).
#
# The postgres branch of ports/common/application/build-virtual.sh, with one
# step added between source preparation and configure, and the port scripts
# unchanged:
#   1. build-domain.sh with PGSU_ONLY="musl runtime": musl, the SDK runtime and
#      the patched source exactly as the port prepares it (0001-0017 for stock;
#      0001-0016 and memory-contexts 0003 for protected). Its link step then
#      stops on the missing objects; that exit is expected and checked.
#   2. hooks-<arm>.patch into src/backend/utils/mmgr/mcxt.c, with
#      exp-reuse-gap.h and reuse-gap-observer.h beside it.
#   3. build-domain.sh with PGSU_FROM=configure and CPPFLAGS=-DEXP_REUSE_GAP
#      (configure takes CPPFLAGS from the environment into Makefile.global).
#   4. build.py, the same relink build-virtual.sh runs.
set -euo pipefail
ARM=${1:?usage: build.sh <stock|protected> <out>}
OUT=$(realpath -m "${2:?usage: build.sh <stock|protected> <out>}")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
PORTS=$REPO/capstone/ports
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
case "$ARM" in
  stock) unset PGSU_NESTED; NESTED=none ;;
  protected) export PGSU_NESTED=sublet; NESTED=postgres ;;
  *) echo "arm is stock or protected, not $ARM" >&2; exit 2 ;;
esac
unset PGSU_GAP_OBSERVER PGSU_FROM PGSU_ONLY CPPFLAGS
mkdir -p "$OUT/source"
export CAPSTONE_APPLICATION_PROFILE=virtual PG_SU_ROOT=$OUT/source
SRC=$PG_SU_ROOT/domain/postgresql-17.5
log() { echo "[reuse-hooks] $*"; }

# 1. musl, runtime, patched source.
set +e
PGSU_ONLY="musl runtime" bash "$PORTS/postgres/app/build-domain.sh" > "$OUT/prepare.log" 2>&1
rc=$?
set -e
[[ $rc != 0 ]] || { echo "prepare pass linked unexpectedly" >&2; exit 2; }
[[ -f $SRC/src/backend/utils/mmgr/mcxt.c && -x $PG_SU_ROOT/runtime/capstone-cc ]] \
  || { tail -20 "$OUT/prepare.log" >&2; echo "prepare pass did not finish (rc $rc)" >&2; exit 2; }
[[ $(cat "$PG_SU_ROOT/domain/nested.mode") == "${PGSU_NESTED:-none}" ]] \
  || { echo "prepared source is not the $ARM arm" >&2; exit 2; }
[[ ! -f $SRC/config.status ]] || { echo "prepare pass configured" >&2; exit 2; }
log "prepared $ARM source (prepare pass rc $rc, expected nonzero)"

# 2. the hooks.
cp "$HERE/exp-reuse-gap.h" "$REPO/capstone/experiments/study/reuse-gap-observer.h" \
  "$SRC/src/backend/utils/mmgr/"
(cd "$SRC" && patch --batch --forward --fuzz=0 -p1 -s < "$HERE/hooks-$ARM.patch") \
  || { echo "hooks-$ARM.patch did not apply" >&2; exit 2; }
log "applied hooks-$ARM.patch"

# 3. configure, make, link.
CPPFLAGS=-DEXP_REUSE_GAP PGSU_FROM=configure bash "$PORTS/postgres/app/build-domain.sh"
grep -q '^CPPFLAGS = .*-DEXP_REUSE_GAP' "$SRC/src/Makefile.global" \
  || { echo "configure dropped CPPFLAGS" >&2; exit 2; }

# 4. the relink of build-virtual.sh.
python3 "$PORTS/common/application/build.py" --app postgres --root "$PG_SU_ROOT" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 1048576 --out "$OUT/image"
grep -c -a 'REUSE-GAP contexts' "$OUT/image/postgres.dom" > /dev/null \
  || { echo "image lacks the reuse report" >&2; exit 2; }
sha256sum "$SRC/src/backend/utils/mmgr/mcxt.c" "$OUT/image/postgres.dom"
