#!/bin/bash
# C-81 exposure, B5's mruby.dom (record a7ac283d7050a5d8, built 10-15 18:56 from the tree of 7049b13d969e with dev147):
# plain rebuild from that tree plus the inert B0_PLUGIN_EXTRA hook (must equal the record), then the libcall twin
# (B0_PLUGIN_EXTRA at the LTO link; rake's bitcode is unchanged), then scan-memcpy-expansion.py --diff. Nothing is run.
main() {
  local TREE=/tmp/capstone/c81/mruby/tree T=/tmp/capstone/c81/mruby W=/tmp/capstone/c81/mruby/w
  export CAPSTONE_LLVM_BIN=${DEV147_BIN:?set DEV147_BIN} B0_MUSL_ARCHIVE=/tmp/capstone/b0/musl-gpct-lto/libc-capstone.a
  export B5_WORK=$W B5_JOBS=8
  rm -rf "$W" "$T/plain" "$T/libcall"; mkdir -p "$W/src"
  cp -a /tmp/capstone/b5m/src/mruby "$W/src/mruby" && rm -rf "$W/src/mruby/build"
  # The script's patch-set stamp hashes `sha256sum` output, which carries the absolute patch path: identical patches
  # (content hash checked below) in another tree read as "another patch set". Re-stamp for this tree's path.
  [ "$(cd /tmp/capstone/b5m/src/../.. && :; sha256sum /tmp/capstone/wt-b0/capstone/ports/mruby/app/patches/4.0.0-rc2/*.patch | cut -c1-64 | sha256sum)" = \
    "$(sha256sum "$TREE"/capstone/ports/mruby/app/patches/4.0.0-rc2/*.patch | cut -c1-64 | sha256sum)" ] || { echo "patch contents differ"; return 1; }
  sha256sum "$TREE"/capstone/ports/mruby/app/patches/4.0.0-rc2/*.patch | sha256sum | cut -d' ' -f1 > "$W/src/mruby/.b5-patchset"
  OUT_DIR=$T/plain nice -n 10 bash "$TREE/capstone/runtime/silicon/build-b5-mruby.sh" > "$T/plain.log" 2>&1; echo "plain rc=$?"
  OUT_DIR=$T/libcall B0_PLUGIN_EXTRA=--plugin-opt=-capstone-lower-memops-via-libcall \
    nice -n 10 bash "$TREE/capstone/runtime/silicon/build-b5-mruby.sh" > "$T/libcall.log" 2>&1; echo "libcall rc=$?"
  echo "plain $(sha256sum "$T/plain/mruby.dom" | cut -c1-16) record a7ac283d7050a5d8 libcall $(sha256sum "$T/libcall/mruby.dom" | cut -c1-16)"
  python3 /tmp/capstone/c81/scan-memcpy-expansion.py --objdump /tmp/capstone/virt-stack/llvm-build/bin/llvm-objdump \
    --clang /tmp/capstone/llvm-dev-current-20261017/bin/clang --diff "$T/plain/mruby.dom" "$T/libcall/mruby.dom" > "$T/diff-mruby.txt" 2>&1
  echo "diff rc=$? $(grep -o 'only in A: [0-9]*   only in B: [0-9]*' "$T/diff-mruby.txt") $(grep -o 'A-only by unit: .*' "$T/diff-mruby.txt")"
  echo TWIN_MRUBY_DONE
}
main "$@"; exit
