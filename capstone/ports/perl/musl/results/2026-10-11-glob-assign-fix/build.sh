#!/usr/bin/env bash
# Glob-assign fix (port patch 0011): stock and protected virtual Perl 5.36.3 from ~/sa-glob-pf/repo
# (16d2171a3edc + 0011), and two native builds of the pristine tarball with the port's native
# Configure line, one with 0011 alone and one without; all four in parallel.
set -u
K=$HOME/pg-virtual-kit M=$HOME/sa-glob-pf R=$M/repo
export LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_LLVM_BUILD_DIR=$K/toolchain CAPSTONE_REPO_ROOT=$R \
  CAPSTONE_TMP_ROOT=$M/tmp PERLD_PROFILE=virtual PERL_MIRROR=$K/inputs/perl-mirror \
  PERL_CROSS_MIRROR=$K/inputs/perl-cross JOBS=24
unset CAPSTONE_SDK
TAR=$K/inputs/perl-mirror/perl-5.36.3.tar.gz
P0011=$R/capstone/ports/perl/musl/patches/5.36.3/0011-glob-assign-reads-source-before-leave.patch
date -u +"BUILD start %Y-%m-%dT%H:%M:%SZ (host clock)"
cap() { s=$(date +%s)
  PERLD_SUBLET=$2 PERLD_ROOT=$M/perl-$1 bash $R/capstone/ports/perl/musl/build-perl-domain.sh > $M/build-$1.log 2>&1
  echo "BUILD $1 rc=$? $(( $(date +%s) - s ))s sha256=$(sha256sum $M/perl-$1/src/perl-5.36.3/perl 2>/dev/null | cut -d' ' -f1)"; }
nat() { s=$(date +%s); N=$M/native-$1; rm -rf $N; mkdir -p $N
  ( set -e; tar xzf $TAR -C $N; chmod -R u+w $N/perl-5.36.3; cd $N/perl-5.36.3
    if [ $1 = with-0011 ]; then patch -p1 --batch --forward --fuzz=0 < $P0011; fi
    ./Configure -des -Dprefix=$N/inst -Uusedl -Uusethreads -Uusemymalloc -Dusenm=false -Dinc_version_list=none
    make -j24 ) > $M/build-native-$1.log 2>&1
  echo "BUILD native-$1 rc=$? $(( $(date +%s) - s ))s sha256=$(sha256sum $N/perl-5.36.3/perl 2>/dev/null | cut -d' ' -f1)"; }
cap stock 0 & cap protected 1 & nat without-0011 & nat with-0011 & wait
echo BUILD-DONE
