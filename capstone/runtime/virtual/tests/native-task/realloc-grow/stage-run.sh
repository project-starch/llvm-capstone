#!/bin/sh
# rgrow: patch 0011's directed test. B = natrealloc SDK, A = nat-fair-cache SDK (control).
S=$(cd "$(dirname "$0")" && pwd)
mkdir -p /tmp/nt && { mountpoint -q /tmp/nt || mount -t tmpfs tmpfs /tmp/nt; } || exit 1
echo "@@SPEC-STAGE rgrow test arms=native commands=4 validations=0"
echo "@@SPEC-ARM-BEGIN native"
cd / && rm -rf /tmp/nt/sr /tmp/nt/b && mkdir -p /tmp/nt/sr /tmp/nt/b && cp "$S"/bin/native/* /tmp/nt/b/ && chmod 755 /tmp/nt/b/* && cd /tmp/nt/sr && mkdir -m 0700 .capstone-tmp || exit 1
n=0
for c in "rgrowB ok" "rgrowA ok" "rgrowB stale" "rgrowA stale"; do
  set -- $c
  echo "@@SPEC-BEGIN $n run native $n"
  "/tmp/nt/b/$1_base.native" $2 > out.$n 2>&1
  rc=$?
  echo "@@RGROW $n $1 $2 rc=$rc out=[$(tr '\n' ' ' < out.$n | cut -c1-200)]"
  echo "@@SPEC-END $n run native $n rc=0"
  n=$((n+1))
done
echo "@@SPEC-ARM-END native"
cd / && rm -rf /tmp/nt/sr /tmp/nt/b
echo "@@SPEC-DONE rgrow test"
