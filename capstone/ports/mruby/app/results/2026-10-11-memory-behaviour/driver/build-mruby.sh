#!/usr/bin/env bash
# mruby 4.0.0-rc2: one virtual SDK, two copies of it, the stock and MRBD_SUBLET=1 builds in parallel.
set -u
. $HOME/sa-mem-mm/env.sh
B=$S/mruby; mkdir -p $B
s=$(date +%s)
(
  set -e
  source $R/tests/capstone-test-env.sh >/dev/null
  export CAPSTONE_APPLICATION_PROFILE=virtual MUSL_CACHE_ROOT=$B/musl-src JOBS=16
  mkdir -p $MUSL_CACHE_ROOT
  MUSL=$(bash $R/ports/musl-capstone/prepare-musl-capstone.sh | tail -1)
  OUT_DIR=$B/musl-build bash $R/ports/musl-capstone/build-musl-capstone.sh
  bash $R/ports/common/application/build-sdk.sh $B/sdk "$MUSL" $B/musl-build/libc-capstone.a \
    -DCAPSTONE_APPLICATION_ARENA_BYTES=67108864
) > $B/sdk.log 2>&1
rc=$?; echo "MRUBY-SDK rc=$rc $(( $(date +%s) - s ))s" > $S/mruby-build.status
[ $rc = 0 ] || exit 1
cp -a $B/sdk $B/sdk-stock; cp -a $B/sdk $B/sdk-prot
arm() {   # name, MRBD_SUBLET
  local t=$(date +%s)
  MRBD_PIN=4.0.0-rc2 MRBD_SDK=$B/sdk-$1 MRBD_ROOT=$B/$1 MRBD_SUBLET=$2 JOBS=16 \
    bash $R/ports/mruby/app/build-mruby-domain.sh > $B/$1.log 2>&1
  echo "MRUBY-BUILD $1 rc=$? $(( $(date +%s) - t ))s" >> $S/mruby-build.status
}
arm stock 0 & arm prot 1 & wait
echo "MRUBY-ALL $(( $(date +%s) - s ))s" >> $S/mruby-build.status
