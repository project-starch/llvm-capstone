#!/usr/bin/env bash
# provenance.sh RUN_DIR ARM IMAGE: provenance.txt for one run-6 PostgreSQL reuse run (run-5's format).
set -euo pipefail
F=$HOME/asr-work/fix-postgres TREE=$HOME/asr-work/fix-postgres/tree-38b1e0a
RUN=$1 ARM=$2 IMAGE=$3 WL=$HOME/asr-work/bench-postgres-perl/gen/s1-t4000 K=$HOME/asr-work/eval/kit-run-3
H=$TREE/capstone/experiments/study/reuse-hooks/postgres
tree_sha() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) | sha256sum | cut -d" " -f1; }
{
  echo "PostgreSQL 17.5 reuse run, pgbench TPC-B-like workload, arm $ARM, $(basename $RUN) (run-6, port layout fix)"
  echo "pre-registration: capstone/docs/plans/2026-11-15-postgres-port-layout-fix.md (branch fix/postgres-port-layout, 7d7d1e0fd3c)"
  echo "tree: export of fix/postgres-port-layout 38b1e0a3529 at $TREE (local snapshot commit $(git -C $TREE rev-parse --short HEAD))"
  echo "platform (kit-run-3):"
  (cd $K && sha256sum qemu-system-riscv64 images/Image images/fw_jump.elf images/initramfs.cpio.gz adapter/capstone-vexec)
  echo "platform sources: qemu-1b $(sha256sum ~/asr-work/q1b/bin/qemu-1b | cut -d" " -f1), Image-0133b9e21cd $(sha256sum ~/asr-work/eval/kernel/Image-0133b9e21cd | cut -d" " -f1), f-keep fw_jump.elf $(sha256sum ~/asr-work/del-fw/out/f-keep/fw_jump.elf | cut -d" " -f1)"
  echo "image:"
  sha256sum $IMAGE $RUN/stage-1/bin/postgres-$ARM.dom
  echo "build inputs:"
  sha256sum $F/tmp/pg-mmgr-host/pg.tar.bz2 $H/build.sh $H/hooks-$ARM.patch $H/exp-reuse-gap.h $H/postgres-17.5-private-shmem.patch $H/postgres-17.5-no-interval-timer.patch $TREE/capstone/experiments/study/reuse-gap-observer.h $TREE/capstone/ports/postgres/memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch
  echo "workload ($WL):"
  sha256sum $WL/tpcb.sql $WL/native.out $WL/manifest.txt $RUN/stage-1/tpcb.sql
  echo "cluster tree $(tree_sha $WL/cluster), staged $(tree_sha $RUN/stage-1/pgcluster)"
  echo "share tree $(tree_sha $RUN/stage-1/share) (from ~/asr-work/eval/prebuilt/reuse/pg-share)"
  echo "harness:"
  sha256sum $TREE/capstone/runtime/virtual/trap_platform.py $H/run-pgbench-reuse.py $H/make-pgbench-workload.sh $H/collect.py $RUN/stage-1/gate.sh
  grep "^RUN $(basename $RUN) " $F/logs/runs.txt | tail -n 1 | sed "s/^/wall time (run-reuse.sh): /"
  echo "workload manifest:"
  sed "s/^/  /" $WL/manifest.txt
} > $RUN/provenance.txt
