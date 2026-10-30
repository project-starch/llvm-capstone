#!/usr/bin/env bash
# SPEC train images with the FIXED heap (dev 348d9cf2c29c + de7cdacda737; SDKs perf/sdk-heapfix-T{1,3} via
# spec/sdks-heapfix) and the controlled code layout (spec-layout c1b3cc5d7c7a, CAPSTONE_SPEC_LAYOUT=heap-last).
# New tree spec/tree-hfl, record spec/rec-hfl. Then runcpu --fake --action=run for train and test, per arm, through
# the layout view ($REC/layout-sdk/<arm>), which make-stage.py reads. Marker: spec/.build-hfl-done; log build-hfl.log.
main() {
  S=${CAPSTONE_WORK:?}/spec; K=${CAPSTONE_WORK:?}/wt-spec-layout/capstone/ports/spec-cpu2017; L=$S/build-hfl.log
  T=$S/tree-hfl; R=$S/rec-hfl
  pin="taskset -c 16-31,48-63 nice -n 10 ionice -c3"
  B="500.perlbench_r 505.mcf_r 519.lbm_r 525.x264_r 538.imagick_r 544.nab_r 557.xz_r"
  [ -e $T ] || [ -e $R ] && { echo "tree-hfl or rec-hfl exists; refusing" > $L; touch $S/.build-hfl-done; exit 1; }
  mkdir -p $T && $pin cp -a $S/pristine/SPEC17 $T/ || { echo "copy FAILED" > $L; touch $S/.build-hfl-done; exit 1; }
  export CAPSTONE_SPEC_PACKAGES=$S BENCHMARKS="$B" CAPSTONE_LLVM_BIN=${CAPSTONE_WORK:?}/cc-prmerged/bin JOBS=12
  export CAPSTONE_SPEC_LAYOUT=heap-last
  unset CAPSTONE_SDK
  $pin bash $K/tools/build.sh $K $T/SPEC17 $S/sdks-heapfix $R capstone-spatial capstone-lifetimes > $L 2>&1
  rc=$?; echo "build rc=$rc" >> $L
  [ $rc = 0 ] || { touch $S/.build-hfl-done; exit 1; }
  cd $T/SPEC17 || exit 1; set +u; source ./shrc > /dev/null
  for size in train test; do
    for arm in capstone-spatial capstone-lifetimes; do
      $pin runcpu --config=capstone-virtual --tune=base --define sdk=$R/layout-sdk/$arm \
        --define kit=$K --define arm_label=$arm --fake --action=run --size=$size --noreportable --iterations=1 $B \
        > $S/fake-hfl-$arm-$size.log 2>&1; echo "fake $arm $size rc=$?" >> $L
    done
  done
  touch $S/.build-hfl-done
}
main "$@"; exit
