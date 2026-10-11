#!/usr/bin/env bash
# Native gate for port patch 0011: the upstream test files that cover S_glob_assign_glob, run by
# upstream's own t/harness (-v, so every TAP line is kept) in the two native build trees, pristine
# and pristine + 0011, built with the port's native Configure line; and the same-answers native
# reference (t/base raw and smoke.pl at 2,000 objects) on both.
set -u
M=$HOME/sa-glob-pf R=$M/repo
SMOKE=$R/capstone/ports/perl/musl/scripts/smoke.pl
BASE="base/cond.t base/if.t base/lex.t base/num.t base/pat.t base/rs.t base/term.t base/translate.t base/while.t"
one() { v=$1 N=$M/native-$1/perl-5.36.3 O=$M/native-gate/$1; rm -rf $O; mkdir -p $O/harness $O/raw
  (cd $N && make test-prep > $O/test-prep.log 2>&1) || echo "test-prep $v rc=$?"
  FILES="op/gv.t op/taint.t op/magic.t $(cd $N/t && ls mro/*.t | tr '\n' ' ') $BASE"
  for f in $FILES; do n=$(echo $f | tr / _); s=$(date +%s.%N)
    (cd $N/t && ./perl harness -v $f > $O/harness/$n.txt 2>&1); rc=$?
    echo "HARNESS $v $f rc=$rc wall=$(echo "$(date +%s.%N) - $s" | bc)"; done
  # same-answers reference: a copy of t/, lib linked, each base file as `perl base/X.t`, then smoke
  RT=$O/rawtree; rm -rf $RT; mkdir -p $RT; cp -a $N/t $RT/t; ln -s $N/lib $RT/lib
  for f in $BASE; do n=$(echo $f | tr / _)
    (cd $RT/t && PERL5LIB=$RT/lib $N/perl $f > $O/raw/$n.stdout 2> $O/raw/$n.stderr); echo "RAW $v $f rc=$?"; done
  mkdir -p $O/smoke; cp $SMOKE $O/smoke/smoke.pl; s=$(date +%s.%N)
  (cd $O/smoke && CAPSTONE_PERL_SMOKE_OBJECTS=2000 $N/perl smoke.pl $O/smoke > $O/raw/smoke.stdout 2> $O/raw/smoke.stderr)
  echo "RAW $v smoke rc=$? wall=$(echo "$(date +%s.%N) - $s" | bc) sha256=$(sha256sum < $O/raw/smoke.stdout | cut -d' ' -f1)"
  ( set -e; cd $N/t; ./perl -e '$x = *foo; *x = $x; print "done\n"' ) > $O/raw/control.stdout 2>&1
  echo "RAW $v control rc=$? out=$(cat $O/raw/control.stdout)"; }
date -u +"GATE start %Y-%m-%dT%H:%M:%SZ"
one without-0011 > $M/native-gate-without-0011.log 2>&1 &
one with-0011 > $M/native-gate-with-0011.log 2>&1 &
wait
date -u +"GATE-DONE %Y-%m-%dT%H:%M:%SZ"
