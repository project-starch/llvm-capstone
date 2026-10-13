#!/usr/bin/env bash
# Perl row of the nested-lifetime controls (docs/plans/2026-10-11-nested-lifetime-controls.md):
# link the controls against each arm's libperl.a, then run both arms at once, one VM each,
# each with its own QEMU lock. Perl builds: build.sh beside this file (from ~/ctl-perl/repo).
set -u
K=$HOME/pg-virtual-kit M=$HOME/ctl-perl R=$M/repo
C=$R/capstone/bug-corpora/perl/release-differential/controls
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain PYTHONDONTWRITEBYTECODE=1
TAG=$(date +%Y%m%d-%H%M%S)
mkdir -p $M/img
for arm in release protected; do
  src=stock; [ $arm = protected ] && src=protected
  bash $C/build-controls.sh $M/perl-$src $M/img/ctl-$arm.dom > $M/img/build-$arm.log 2>&1
  echo "LINK $arm rc=$? $(sha256sum $M/img/ctl-$arm.dom | cut -d' ' -f1)"
done
date -u +"RUN start %Y-%m-%dT%H:%M:%SZ (host clock) tag=$TAG"
for arm in release protected; do
  ( CAPSTONE_QEMU_LOCK=$M/qemu-$arm.lock bash $C/run-controls.sh $arm $M/img/ctl-$arm.dom $K/kit-sublet \
      $M/guest-$TAG-$arm > $M/log-$TAG-$arm.txt 2>&1; echo "RUN $arm rc=$?" ) &
done
wait
date -u +"RUN done %Y-%m-%dT%H:%M:%SZ"
python3 $C/collect.py $M/results-$TAG.json --llvm-bin $K/toolchain/bin \
  --arm release $M/guest-$TAG-release/serial.log $M/img/ctl-release.dom \
  --arm protected $M/guest-$TAG-protected/serial.log $M/img/ctl-protected.dom \
  --meta tag=$TAG --meta snapshot=$(cat $M/SNAPSHOT_REV)
echo "COLLECT rc=$? $M/results-$TAG.json"
