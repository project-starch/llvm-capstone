#!/usr/bin/env bash
# The Perl release-differential corpus on the 0011 images with the recorded runner
# (runners/virtual/run-virtual.py, unchanged at 16d2171a3edc): stock on vmA, protected on vmB;
# then qualify.py on the native-gate files (op/gv.t first) on the same VMs.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf R=$M/repo OUT=$M/corpus
export LD_LIBRARY_PATH=$K/toolchain/hostlib PYTHONDONTWRITEBYTECODE=1
mkdir -p $OUT
C=$R/capstone/bug-corpora/perl/release-differential/runners/virtual
Q="op/gv.t op/taint.t op/magic.t $(cd $M/perl-stock/src/perl-5.36.3/t && ls mro/*.t | tr '\n' ' ')"
arm() { s=$(date +%s); python3 $C/run-virtual.py $OUT/$1 --arm $1 --state $M/$3 --perl $M/perl-$2 \
    --llvm-bin $K/toolchain/bin > $OUT/$1.log 2>&1; echo "ARM $1 rc=$? $(( $(date +%s) - s ))s"
  s=$(date +%s); python3 $C/qualify.py $OUT/qualify-$2 --state $M/$3 --perl $M/perl-$2 --llvm-bin $K/toolchain/bin \
    $Q > $OUT/qualify-$2.log 2>&1; echo "QUALIFY $2 rc=$? $(( $(date +%s) - s ))s"; }
date -u +"CORPUS start %Y-%m-%dT%H:%M:%SZ"
arm virtual-malloc stock vmA & arm virtual-nested-pools protected vmB & wait
date -u +"CORPUS-DONE %Y-%m-%dT%H:%M:%SZ"
