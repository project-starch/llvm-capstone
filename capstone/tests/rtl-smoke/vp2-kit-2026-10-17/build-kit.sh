#!/bin/bash
# vp2 acceptance kit, Tiers 1 and 2: the virtual SDK, the adapter (capstone_vm.ko + capstone-vexec) for BOTH kernels (the
# QEMU guest's 6.1.0 for the prediction, the board's 6.4.14 for the board), contract.dom, SQLite's sqlite3.dom and mruby,
# from one runtime revision; then the board gates and the QEMU predictions.
#   bash build-kit.sh <runtime rev> <label>      e.g. origin/dev dev-proof; later the compiler lane's ABI v6 branch
# Status lines to <out>/status; ends with KIT-DONE. Reference QEMU: capstone-qemu virtual-capstone d35af47b163c.
# Inputs, all required: KIT_REPO (the main clone, with its buildroot submodules: a worktree has none), KIT_LLVM (an LLVM
# build with -capstone-image-gp), KIT_QEMU_REF (a d35af47b163c checkout with build/qemu-system-riscv64), KIT_IMAGES (the
# guest's Image, fw_jump.elf and rootfs.ext2), KIT_MUSL_TARBALL (musl-1.2.5.tar.gz). KIT_OUT is the output root.
main() {
  set -u
  local REV=${1:?runtime rev} LABEL=${2:?label}
  local R=${KIT_REPO:?} O=${KIT_OUT:-/tmp/capstone/vp2-prep}/$LABEL QREF=${KIT_QEMU_REF:?} IMGS=${KIT_IMAGES:?}
  local BQ=$R/capstone/caplifive-buildroot BF=$R/capstone/caplifive-system/sw/buildroot/build-fpga
  mkdir -p $O; local ST=$O/status; : > $ST
  log(){ echo "$* $(date +%T)" | tee -a $ST; }
  git -C $R fetch -q origin 2>/dev/null
  local SHA; SHA=$(git -C $R rev-parse --short=12 "$REV") || { log "FAIL rev $REV"; return 1; }
  # the source: a detached worktree, with the reference QEMU tree as its capstone-qemu (build-adapter.sh reads the
  # revocation-table ABI header from there, so the module and the emulator agree)
  local S=$O/src
  [ -d $S ] || git -C $R worktree add -q --detach $S $SHA || { log "FAIL worktree"; return 1; }
  [ "$(git -C $S rev-parse --short=12 HEAD)" = "$SHA" ] || { log "FAIL worktree is not at $SHA"; return 1; }
  [ -L $S/capstone/capstone-qemu ] || { rmdir $S/capstone/capstone-qemu 2>/dev/null; ln -s $QREF $S/capstone/capstone-qemu; }
  log "SRC $REV = $SHA, capstone-qemu -> $(git -C $QREF rev-parse --short=12 HEAD)"
  local dirty; dirty=$(git -C $S status --short --untracked-files=no | grep -v ' capstone/capstone-qemu$' | tr '\n' ' ')
  [ -n "$dirty" ] && log "SRC-LOCAL-CHANGES $dirty (git -C $S diff)"
  # the compiler: an existing virtual-capable LLVM build (the virtual profile needs -capstone-image-gp, 5d077ae20a20;
  # dev147 predates it). The October 10 build bd372e25f5fb fails dev's runtime (C-80); use a build carrying 7ce213020f87.
  local L=${KIT_LLVM:?}
  [ -x $L/bin/clang ] || { log "FAIL no clang in $L"; return 1; }
  ln -sfn $L $O/llvm-build
  local igp; igp=$(strings $O/llvm-build/lib/libLLVMCapstoneCodeGen.so | grep -c capstone-image-gp)
  [ "$igp" -ge 1 ] || { log "FAIL the built compiler lacks -capstone-image-gp"; return 1; }
  export CAPSTONE_LLVM_BUILD_DIR=$O/llvm-build CAPSTONE_LLVM_BIN=$O/llvm-build/bin
  export CAPSTONE_CLANG=$CAPSTONE_LLVM_BIN/clang CAPSTONE_LD_LLD=$CAPSTONE_LLVM_BIN/ld.lld CAPSTONE_TMP_ROOT=$O/tmp JOBS=12
  export CAPSTONE_BUILDROOT_DIR=$BQ
  # run.py takes the host-wide QEMU lock itself (flock), from this variable (capstone-test-env.sh:79)
  export CAPSTONE_QEMU_LOCK=${CAPSTONE_QEMU_LOCK:-$HOME/.capstone-locks/qemu.lock}
  mkdir -p $O/tmp $O/musl-cache; cp -n ${KIT_MUSL_TARBALL:?} $O/musl-cache/
  export MUSL_CACHE_ROOT=$O/musl-cache
  log "compiler: $($CAPSTONE_CLANG --version | head -1)"
  local musl; musl=$(bash $S/capstone/ports/musl-capstone/prepare-musl-capstone.sh 2> $O/musl.log | tail -1)
  [ -d "$musl" ] || { log "FAIL-MUSL (musl.log)"; return 1; }
  nice -n 10 bash $S/capstone/runtime/virtual/build-sdk.sh $O/out "$musl" > $O/sdk.log 2>&1 || { log "FAIL-SDK (sdk.log)"; return 1; }
  log "SDK-DONE"
  KERNEL_BUILD=$BQ/build/build/linux-custom CROSS_COMPILE=$BQ/build/host/bin/riscv64-buildroot-linux-gnu- \
    nice -n 10 bash $S/capstone/runtime/virtual/build-adapter.sh $O/out/adapter-qemu > $O/adapter-qemu.log 2>&1 || { log "FAIL-ADAPTER-QEMU"; return 1; }
  # the board's module: the FPGA kernel tree, and buildroot's external.mk flags (rv64g, no RVC: a compressed module
  # hangs the board in insmod, ISSUES/memory 'out-of-tree package build loses EXTRA_CFLAGS')
  KERNEL_BUILD=$BF/build/linux-6.4.14 CROSS_COMPILE=$BF/host/bin/riscv64-buildroot-linux-gnu- KCFLAGS="-march=rv64g -mabi=lp64d" \
    nice -n 10 bash $S/capstone/runtime/virtual/build-adapter.sh $O/out/adapter-board > $O/adapter-board.log 2>&1 || { log "FAIL-ADAPTER-BOARD"; return 1; }
  log "MODULE-HEADERS $(cd $O/out/adapter-board/module && sha256sum *.h | cut -c1-16,65- | tr '\n' ' ')"
  log "ADAPTERS-DONE qemu-ko=$(sha256sum $O/out/adapter-qemu/module/capstone_vm.ko | cut -c1-16) board-ko=$(sha256sum $O/out/adapter-board/module/capstone_vm.ko | cut -c1-16) board-vexec=$(sha256sum $O/out/adapter-board/capstone-vexec | cut -c1-16)"
  # the corpus runners' kit (bug-corpora/tools/virtualvm.py FILES): the reference QEMU, the guest images, the QEMU adapter
  local K=$O/out/kit; rm -rf $K; mkdir -p $K/images $K/adapter/module
  cp $QREF/build/qemu-system-riscv64 $K/ && cp /lib/x86_64-linux-gnu/libslirp.so.0 $K/ \
    && cp $IMGS/{Image,fw_jump.elf,rootfs.ext2} $K/images/ \
    && cp $O/out/adapter-qemu/capstone-vexec $O/out/adapter-qemu/capstone-job $K/adapter/ && cp $O/out/adapter-qemu/module/capstone_vm.ko $K/adapter/module/ \
    || { log "FAIL-KIT"; return 1; }
  log "KIT-ASSEMBLED $K qemu=$(sha256sum $K/qemu-system-riscv64 | cut -c1-16) rootfs=$(sha256sum $K/images/rootfs.ext2 | cut -c1-16)"
  $O/out/sdk/capstone-cc -O1 -I$S/capstone/runtime/include $S/capstone/runtime/virtual/contract.c -o $O/out/contract.dom > $O/contract.log 2>&1 || { log "FAIL-CONTRACT"; return 1; }
  log "CONTRACT-DONE $(sha256sum $O/out/contract.dom | cut -c1-16)"
  # Tier 2: the run.py gate's two application images, with this SDK
  ( cd $S && bash capstone/ports/sqlite/app/prepare-sources.sh $O/out/sqlite-source && bash capstone/ports/sqlite/app/build-domain.sh $O/out/sdk $O/out/sqlite-source $O/out/sqlite ) > $O/sqlite.log 2>&1 || log "FAIL-SQLITE (sqlite.log)"
  ( cd $S && MRBD_ROOT=$O/out/mruby MRBD_SDK=$O/out/sdk MRBD_PIN=4.0.0-rc2 bash capstone/ports/mruby/app/build-mruby-domain.sh ) > $O/mruby.log 2>&1 || log "FAIL-MRUBY (mruby.log)"
  log "APPS sqlite=$(sha256sum $O/out/sqlite/sqlite3.dom 2>/dev/null | cut -c1-16) mruby=$(sha256sum $O/out/mruby/src/mruby/build/capstone/bin/mruby 2>/dev/null | cut -c1-16)"
  # board gates on the board's module and launcher
  local KO=$O/out/adapter-board/module/capstone_vm.ko OD=$BF/host/bin/riscv64-buildroot-linux-gnu-objdump
  local rvc; rvc=$($OD -d $KO | python3 -c "import sys,re; print(sum(1 for l in sys.stdin if re.match(r'^\s+[0-9a-f]+:\s+[0-9a-f]{4}\s+\S', l)))")
  local vm; vm=$($BF/host/bin/riscv64-buildroot-linux-gnu-readelf -p .modinfo $KO 2>/dev/null | grep -o 'vermagic=[^ ]*' | head -1)
  log "BOARD-GATES ko 16-bit instructions=$rvc (must be 0); $vm (kernel $(cat $BF/build/linux-6.4.14/include/config/kernel.release)); vexec: $(file -b $O/out/adapter-board/capstone-vexec | cut -c1-60)"
  # the QEMU prediction: run.py's gate on the QEMU kit (the guest's own kernel and adapter)
  if [ -n "${KIT_SKIP_RUNPY:-}" ]; then log "RUNPY skipped (KIT_SKIP_RUNPY)"; log KIT-DONE; return 0; fi
  python3 $S/capstone/runtime/virtual/run.py --qemu $QREF/build/qemu-system-riscv64 --images $IMGS \
    --adapter $O/out/adapter-qemu --application $O/out/contract.dom --sqlite $O/out/sqlite/sqlite3.dom \
    --mruby $O/out/mruby/src/mruby/build/capstone/bin/mruby --record $O/out/result.json --timeout ${RUNPY_TIMEOUT:-1500} > $O/runpy.log 2>&1
  log "RUNPY rc=$? record=$(ls $O/out/result.json 2>/dev/null)"
  log KIT-DONE
}
main "$@"; exit $?
