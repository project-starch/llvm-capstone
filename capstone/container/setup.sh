#!/usr/bin/env bash
# One-shot bring-up of the Capstone toolchain, QEMU and the guest image.
# RUN IT THROUGH run.sh, never directly on the host:
#
#   ./run.sh capstone/container/setup.sh              # everything
#   ./run.sh capstone/container/setup.sh llvm qemu    # selected stages
#   FORCE_CONFIGURE=1 ./run.sh capstone/container/setup.sh llvm
#
# Stages: submodules  llvm  qemu  buildroot
# Every stage is idempotent -- re-running skips work that is already done, so an
# interrupted setup is resumed by re-issuing the same command.
set -euo pipefail

ROOT=${CAPSTONE_REPO_ROOT:?must run through run.sh}
cd "$ROOT"
JOBS=${JOBS:-$(nproc)}

# The pin llvm-capstone records for the QEMU submodule. Asserted rather than assumed:
# a silent drift here changes which QEMU the whole test matrix runs against.
QEMU_PIN=deb7d757565dd22f950c565579987e4864891ec9

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

WANT=("$@"); [ ${#WANT[@]} -gt 0 ] || WANT=(submodules llvm qemu buildroot)
want() { for s in "${WANT[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# ---------------------------------------------------------------- submodules
if want submodules; then
  say "submodules"
  # Populating these is the HOST's job -- see capstone/container/fetch-submodules.sh.
  # project-starch/capstone-qemu is private, and the GitHub token that unlocks it is
  # deliberately left outside the container. This stage therefore only CHECKS.
  missing=0
  for m in capstone/capstone-qemu capstone/caplifive-buildroot; do
    if [ -z "$(ls -A "$m" 2>/dev/null)" ]; then echo "  EMPTY   $m"; missing=1
    else echo "  present $m"; fi
  done
  if [ "$missing" = 1 ]; then
    die "submodules are not populated. Run this ON THE HOST first:
       capstone/container/fetch-submodules.sh
     (or re-run with CAPSTONE_GIT_CREDS=1 to mount ~/.git-credentials into the container)"
  fi
  have=$(git -C capstone/capstone-qemu rev-parse HEAD)
  if [ "$have" != "$QEMU_PIN" ]; then
    echo "  NOTE: capstone-qemu is at ${have:0:10}, not the recorded pin ${QEMU_PIN:0:10}"
    echo "        (fine if deliberate; QEMU is built from whatever is checked out here)"
  else
    echo "  capstone-qemu at the recorded pin ${QEMU_PIN:0:10}"
  fi
  # The four caplifive-buildroot submodules qemu_capstone_defconfig needs. Checked
  # individually because the failure is otherwise deferred to the middle of a long
  # Buildroot run: an empty components/linux only surfaces when the kernel package builds.
  for d in buildroot components/linux components/opensbi components/opensbi/lib/sbi/capstone-sbi \
           package/capstone-sbi-domain/capstone-sbi; do
    [ -n "$(ls -A "capstone/caplifive-buildroot/$d" 2>/dev/null)" ] ||
      die "caplifive-buildroot/$d is empty -- run capstone/container/fetch-submodules.sh on the host"
  done
  echo "  caplifive-buildroot submodules present"
  [ -d capstone/capstone-c/src ] ||
    die "capstone/capstone-c is empty -- the OpenSBI monitor cannot be regenerated without it"
  echo "  capstone-c present"
fi

# ---------------------------------------------------------------------- llvm
if want llvm; then
  BUILD=${CAPSTONE_LLVM_BUILD_DIR:-$ROOT/llvm/cmake-build-debug}

  if [ ! -f "$BUILD/CMakeCache.txt" ] || [ -n "${FORCE_CONFIGURE:-}" ]; then
    say "llvm configure -> $BUILD"
    # The directory is named cmake-build-debug but the build type is RelWithDebInfo.
    # The NAME is a contract: it is what CAPSTONE_LLVM_BUILD_DIR defaults to in
    # capstone/tests/capstone-test-env.sh, and what build-toolchain.sh and
    # toolchain-fresh.py fall back to. Renaming it would mean an override in every shell.
    #
    # Capstone is in LLVM_ALL_TARGETS (llvm/CMakeLists.txt), NOT in the experimental
    # list, so no LLVM_EXPERIMENTAL_TARGETS_TO_BUILD is needed. X86 is the host.
    #
    # LLVM_PARALLEL_LINK_JOBS=2 is the memory guard on a 31 GB box: it is the
    # debug-info links, not the compiles, that spike, and this host has no swap.
    cmake -G Ninja -S llvm -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS='clang;lld' \
      -DLLVM_TARGETS_TO_BUILD='Capstone;X86' \
      -DLLVM_USE_SPLIT_DWARF=ON \
      -DLLVM_USE_LINKER=lld \
      -DLLVM_PARALLEL_LINK_JOBS=2 \
      -DLLVM_CCACHE_BUILD=ON \
      -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF
  else
    say "llvm already configured at $BUILD (FORCE_CONFIGURE=1 to redo)"
  fi

  say "llvm build (-j$JOBS) -- the long pole, expect hours on a small host"
  # Exactly the binaries capstone-test-env.sh names, plus the reduction tools the bug
  # workflow uses. NOT a bare `ninja`: building all targets also builds the unit-test
  # binaries, which is a large amount of extra work you only need for `check-llvm`.
  # If you do want the full lit suite:  ninja -C "$BUILD" check-llvm
  ninja -C "$BUILD" -j"$JOBS" \
    clang lld llc llvm-mc llvm-objdump llvm-readobj llvm-nm llvm-reduce llvm-stress

  # llvm-lit is deliberately NOT in that list. $CAPSTONE_LLVM_LIT points at
  # bin/llvm-lit, but that is a python script cmake GENERATES AT CONFIGURE TIME -- there
  # is no ninja target by that name, and asking for one fails the whole build with
  # "ninja: error: unknown target 'llvm-lit'". Assert the file instead.
  [ -x "$BUILD/bin/llvm-lit" ] ||
    die "bin/llvm-lit is missing -- configure did not generate it (LLVM_INCLUDE_TESTS off?)"

  # A clang that built cleanly with the Capstone target accidentally dropped would still
  # print a version banner. Ask llc what it actually registered.
  "$BUILD/bin/llc" --version | grep -qi capstone ||
    die "llc has no Capstone target registered -- LLVM_TARGETS_TO_BUILD did not take"
  echo "  ok: llc registers the Capstone target"
fi

# ---------------------------------------------------------------------- qemu
if want qemu; then
  say "qemu"
  cd "$ROOT/capstone/capstone-qemu"
  # configure.sh is the repo's own script: it makes build/, and runs ../configure with
  # --enable-slirp --enable-debug --target-list=riscv64-softmmu --prefix=../installation.
  # Use it rather than hand-rolling a configure line, so this stays in step with the repo.
  [ -f build/build.ninja ] || [ -f build/Makefile ] || ./configure.sh
  make -C build -j"$JOBS"
  make -C build install
  cd "$ROOT"
  # CAPSTONE_QEMU_BINARY points into the BUILD tree, not installation/, so this is the
  # file the test scripts will actually exec.
  [ -x capstone/capstone-qemu/build/qemu-system-riscv64 ] ||
    die "qemu-system-riscv64 not produced in capstone/capstone-qemu/build"
  capstone/capstone-qemu/build/qemu-system-riscv64 --version | head -1
fi

# ----------------------------------------------------------------- buildroot
if want buildroot; then
  say "buildroot (TARGET=qemu)"
  BR="$ROOT/capstone/caplifive-buildroot"
  CC_PATH=${CAPSTONE_CC_PATH:-$ROOT/capstone/capstone-c}

  # Drive the tree's OWN Makefile, not `make -C buildroot` directly. It is not a wrapper:
  #   * it regenerates the monitor assembly ($(CAPSTONE_S_OUTPUT)) by running capstone-c,
  #     and those .c.S files are gitignored build artifacts that exist nowhere else;
  #   * it passes -DCAPSTONE_TARGET_QEMU -DCAPSTONE_DEBUG_ENABLE, which select the QEMU
  #     variant of the monitor. Building by hand silently omits them;
  #   * it keeps the output in build-qemu/, separate from build-fpga/. That separation is
  #     the fix for ISSUES.md C-11, where a shared build/ let an FPGA fw_jump.o with an
  #     embedded device tree get relinked into the QEMU firmware, which then discarded
  #     QEMU's DTB and hung with no serial output at all.
  [ -d "$CC_PATH/src" ] ||
    die "capstone-c is not populated at $CC_PATH -- run capstone/container/fetch-submodules.sh on the host"

  cd "$BR"
  export BR2_DL_DIR=${BR2_DL_DIR:-/home/builder/buildroot-dl}
  mkdir -p "$BR2_DL_DIR"

  # JOBS is the tree's own knob and defaults to 90 -- written for a 112-core host, and its
  # comment says a full-width link storm has taken that machine down. Pass this box's count.
  MK=(make TARGET=qemu CAPSTONE_CC_PATH="$CC_PATH" JOBS="$JOBS")

  [ -f build-qemu/.config ] || "${MK[@]}" setup
  "${MK[@]}" build

  # build/ is the per-checkout symlink every harness script reads (capstone-test-env.sh's
  # CAPSTONE_BUILDROOT_DIR/build/images/..., capstone/utils/run-qemu.sh). The Makefile
  # deliberately does NOT create it: which target a checkout serves is a property of the
  # checkout, not the tree.
  if [ ! -e build ]; then
    ln -s build-qemu build
    echo "  linked build -> build-qemu"
  elif [ -L build ]; then
    echo "  build -> $(readlink build)"
  else
    die "build/ is a real directory, not the expected symlink to build-qemu.
     If it is left over from a hand-rolled 'make -C buildroot ... O=build', move it aside:
       mv '$BR/build' '$BR/build.obsolete' && ln -s build-qemu '$BR/build'"
  fi

  for f in fw_jump.elf Image rootfs.ext2; do
    [ -f "build/images/$f" ] || die "buildroot finished but build/images/$f is missing"
  done
  echo "  ok: fw_jump.elf, Image, rootfs.ext2 present"
  cd "$ROOT"
fi

# ------------------------------------------------------------------- summary
say "state"
B=${CAPSTONE_LLVM_BUILD_DIR:-$ROOT/llvm/cmake-build-debug}
for p in "$B/bin/clang" "$B/bin/ld.lld" \
         "$ROOT/capstone/capstone-qemu/build/qemu-system-riscv64" \
         "$ROOT/capstone/caplifive-buildroot/build/images/Image"; do
  if [ -e "$p" ]; then printf '  present  %s\n' "$p"; else printf '  MISSING  %s\n' "$p"; fi
done
echo
echo "Next: ./run.sh capstone/container/verify.sh"
