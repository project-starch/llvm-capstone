#!/bin/bash
# C-81 exposure, silicon set 1 (memcached plain-heap, silicon-level0): a plain rebuild (must reproduce the board's hashes:
# the B0_PLUGIN_EXTRA hook changes nothing when empty) and a compile-only twin with the libcall lowering at LTO codegen
# (must differ, else the flag did not apply), then scan-memcpy-expansion.py --diff per image. Nothing here runs an image.
main() {
  local R=/tmp/capstone/wt-b0 C=/tmp/capstone/wt-b0/capstone/bug-corpora/memcached/plain-heap-repros T=/tmp/capstone/c81/set1
  local BIN=${DEV147_BIN:?set DEV147_BIN to the frozen dev147 toolchain bin (clang-22 sha256 504dca34...)}
  export B0_MUSL_ARCHIVE=/tmp/capstone/b0/musl-gpct-lto/libc-capstone.a CAPSTONE_LLVM_BIN=$BIN
  rm -rf $T; mkdir -p $T
  nice -n 10 python3 $R/capstone/bug-corpora/tools/run-silicon-cases.py build --arm silicon-level0 --out $T/plain --jobs 2 $C > $T/plain.log 2>&1
  echo "plain rc=$?"
  B0_PLUGIN_EXTRA=--plugin-opt=-capstone-lower-memops-via-libcall nice -n 10 python3 $R/capstone/bug-corpora/tools/run-silicon-cases.py build \
    --arm silicon-level0 --out $T/libcall --jobs 2 $C > $T/libcall.log 2>&1
  echo "libcall rc=$?"
  echo "== plain vs the board's record (build-silicon-level0.log):"
  diff <(grep ' ok ' /tmp/capstone/corpus-sil/build-silicon-level0.log | awk '{print $1,$3}') <(grep ' ok ' $T/plain.log | awk '{print $1,$3}') && echo "PLAIN == BOARD (all images)"
  echo "== libcall vs plain hashes:"
  paste <(grep ' ok ' $T/plain.log | awk '{print $1,$3}') <(grep ' ok ' $T/libcall.log | awk '{print $3}') | awk '{print $1, $2, $3, ($2==$3?"SAME":"differs")}'
  echo "== diff (expansions present only in the plain build):"
  for d in $T/plain/silicon-level0/build/*/; do
    n=$(basename $d); a=$d/$n.dom; b=$T/libcall/silicon-level0/build/$n/$n.dom
    python3 /tmp/capstone/c81/scan-memcpy-expansion.py --objdump /tmp/capstone/virt-stack/llvm-build/bin/llvm-objdump --clang $BIN/clang --diff $a $b > $T/diff-$n.txt 2>&1
    echo "$n rc=$? $(tail -1 $T/diff-$n.txt | cut -c1-150)"
  done
  echo TWIN_SET1_DONE
}
main "$@"; exit
