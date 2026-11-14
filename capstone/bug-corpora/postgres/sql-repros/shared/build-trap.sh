#!/usr/bin/env bash
# build-trap.sh stock|protected NEW_OUTPUT: the PostgreSQL 17.5 backend of one sql-repros arm on the trap route.
#
# build-virtual.sh's postgres branch (PGSU_NESTED none or sublet), with one step between source preparation
# and configure: the two study patches that let the single-user backend run as a protected program, applied
# identically to both arms (experiments/study/reuse-hooks/postgres/):
#   postgres-17.5-private-shmem.patch      the main segment and dynamic segments as private anonymous
#                                          mappings: a protected program cannot map shared memory, and a
#                                          single-user backend shares them with nobody
#   postgres-17.5-no-interval-timer.patch  a timeout is not enforced when setitimer returns ENOSYS
# The memory contexts the corpus measures are untouched. run-arm.py runs the backend with
# dynamic_shared_memory_type=mmap, the implementation the first patch turns private.
#   stock      virtual-malloc    (nested none)
#   protected  virtual-pg-pools  (nested postgres: the memory-context port's patch 0003)
set -euo pipefail
ARM=${1:?usage: build-trap.sh <stock|protected> <out>}
OUT=$(realpath -m "${2:?usage: build-trap.sh <stock|protected> <out>}")
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/../../../../.." && pwd)
PORTS=$REPO/capstone/ports
STUDY=$REPO/capstone/experiments/study/reuse-hooks/postgres
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
log() { echo "[build-trap] $*"; }

# 1. musl, runtime, patched source; the link step stops on the missing objects, as expected.
set +e
PGSU_ONLY="musl runtime" bash "$PORTS/postgres/app/build-domain.sh" > "$OUT/prepare.log" 2>&1
rc=$?
set -e
[[ $rc != 0 ]] || { echo "prepare pass linked unexpectedly" >&2; exit 2; }
[[ -f $SRC/src/backend/port/sysv_shmem.c && -x $PG_SU_ROOT/runtime/capstone-cc ]] \
  || { tail -20 "$OUT/prepare.log" >&2; echo "prepare pass did not finish (rc $rc)" >&2; exit 2; }
[[ $(cat "$PG_SU_ROOT/domain/nested.mode") == "${PGSU_NESTED:-none}" ]] \
  || { echo "prepared source is not the $ARM arm" >&2; exit 2; }
log "prepared $ARM source (prepare pass rc $rc, expected nonzero)"

# 2. the two study patches, the same in both arms.
for p in postgres-17.5-private-shmem.patch postgres-17.5-no-interval-timer.patch; do
  (cd "$SRC" && patch --batch --forward --fuzz=0 -p1 -s < "$STUDY/$p") || { echo "$p did not apply" >&2; exit 2; }
  log "applied $p"
done

# 3. configure, make, link; 4. the relink of build-virtual.sh.
PGSU_FROM=configure bash "$PORTS/postgres/app/build-domain.sh"
python3 "$PORTS/common/application/build.py" --app postgres --root "$PG_SU_ROOT" \
  --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
  --input-revision HEAD --stack-bytes 1048576 --out "$OUT/image"
sha256sum "$STUDY"/postgres-17.5-private-shmem.patch "$STUDY"/postgres-17.5-no-interval-timer.patch \
  "$OUT/image/postgres.dom" | tee "$OUT/build-trap.sha256"
