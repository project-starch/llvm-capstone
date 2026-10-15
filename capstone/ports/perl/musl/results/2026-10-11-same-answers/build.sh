#!/usr/bin/env bash
# Build stock and protected virtual Perl 5.36.3 (and the native reference) from ~/sa-perl/repo.
set -u
K=$HOME/pg-virtual-kit R=$HOME/sa-perl/repo
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain CAPSTONE_REPO_ROOT=$R \
  CAPSTONE_TMP_ROOT=$HOME/sa-perl/tmp PERLD_PROFILE=virtual \
  PERL_MIRROR=$K/inputs/perl-mirror PERL_CROSS_MIRROR=$K/inputs/perl-cross JOBS=24
for arm in stock protected; do
  s=$(date +%s)
  if [ $arm = protected ]; then sub=1; else sub=0; fi
  PERLD_SUBLET=$sub PERLD_ROOT=$HOME/sa-perl/perl-$arm bash $R/capstone/ports/perl/musl/build-perl-domain.sh > $HOME/sa-perl/build-$arm.log 2>&1
  echo "BUILD $arm rc=$? $(( $(date +%s) - s ))s"
done
echo BUILD-DONE
