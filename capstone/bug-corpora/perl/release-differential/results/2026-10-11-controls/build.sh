#!/usr/bin/env bash
# Nested-lifetime controls, Perl row: stock and protected virtual Perl 5.36.3 from ~/ctl-perl/repo
# (git archive 53325bba0d5c capstone compiler-rt/lib/builtins), in parallel, JOBS=24.
set -u
K=$HOME/pg-virtual-kit M=$HOME/ctl-perl R=$M/repo
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain CAPSTONE_REPO_ROOT=$R \
  CAPSTONE_TMP_ROOT=$M/tmp PERLD_PROFILE=virtual PERL_MIRROR=$K/inputs/perl-mirror \
  PERL_CROSS_MIRROR=$K/inputs/perl-cross JOBS=24
unset CAPSTONE_SDK
date -u +"BUILD start %Y-%m-%dT%H:%M:%SZ (host clock)"
cap() { s=$(date +%s)
  PERLD_SUBLET=$2 PERLD_ROOT=$M/perl-$1 bash $R/capstone/ports/perl/musl/build-perl-domain.sh > $M/build-$1.log 2>&1
  echo "BUILD $1 rc=$? $(( $(date +%s) - s ))s sha256=$(sha256sum $M/perl-$1/src/perl-5.36.3/perl 2>/dev/null | cut -d' ' -f1)"; }
cap stock 0 & cap protected 1 & wait
echo BUILD-DONE
