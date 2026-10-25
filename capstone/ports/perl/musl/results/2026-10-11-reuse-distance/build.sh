#!/usr/bin/env bash
# Reuse-distance builds: Perl and FFmpeg, stock and protected, from ~/sa-reuse-pf/repo, in parallel.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-reuse-pf R=$M/repo
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain CAPSTONE_REPO_ROOT=$R \
  CAPSTONE_TMP_ROOT=$M/tmp PERL_MIRROR=$K/inputs/perl-mirror PERL_CROSS_MIRROR=$K/inputs/perl-cross \
  JOBS=24 FFAPP_JOBS=24
unset CAPSTONE_SDK
date -u +"BUILD start %Y-%m-%dT%H:%M:%SZ (host clock)"
p() { s=$(date +%s); bash $R/capstone/experiments/study/reuse-hooks/perl/build.sh $1 $M/perl-$1 > $M/build-perl-$1.log 2>&1
  echo "BUILD perl-$1 rc=$? $(( $(date +%s) - s ))s"; }
f() { s=$(date +%s); bash $R/capstone/experiments/study/reuse-hooks/ffmpeg/build.sh $1 $M/ffmpeg-$1 $K/inputs/ffmpeg-9.0.1.tar.xz > $M/build-ffmpeg-$1.log 2>&1
  echo "BUILD ffmpeg-$1 rc=$? $(( $(date +%s) - s ))s"; }
p stock & p protected & f stock & f protected & wait
echo BUILD-DONE
