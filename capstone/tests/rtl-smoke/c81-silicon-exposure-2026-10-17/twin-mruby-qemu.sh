#!/bin/bash
# C-81, mruby's open question (the indirect path): run the compile-only libcall twin f87f9cbf under set 10's own
# silicon-mimic QEMU stage (qemu-arm2.sh: qemu-round ea7af742, firmware d5c57ee765c9, GP_FABRICATE=0), the arm built by
# set 10's recipe with --image pointing at the twin. smoke.rb (the arm's control) must pass first: the flag's status for
# RUNNING is "does not work yet". Compared with set 10's own QEMU record, /tmp/capstone/corpus-sil/r17/.../qemu-matrix.json.
main() {
  local O=/tmp/capstone/c81/mruby-q TW=/tmp/capstone/c81/mruby/libcall/mruby.dom
  [ "$(sha256sum $TW | cut -c1-16)" = f87f9cbf0fa8e5a6 ] || { echo "twin is not f87f9cbf"; return 1; }
  rm -rf $O; mkdir -p $O
  CAPSTONE_LLVM_BIN=${DEV147_BIN:?} B0_MUSL_ARCHIVE=/tmp/capstone/b0/musl-gpct-lto/libc-capstone.a nice -n 10 python3 \
    /tmp/capstone/wt-b0/capstone/bug-corpora/tools/run-silicon-cases.py build --arm silicon-mruby-bounds --out $O --jobs 1 \
    --image $TW /tmp/capstone/wt-b0/capstone/bug-corpora/mruby/release-differential > $O/build.log 2>&1
  echo "build rc=$? $(tail -1 $O/build.log)"
  # capstone-vm takes the global lock itself (LOCK_NB): retry the whole stage, bounded, only while another VM owns it.
  local end=$(( $(date +%s) + 1800 )) rc
  while :; do
    OUTROOT=$O VMTAG=-c81 CDERIVE_ROUND=1 timeout 5400 bash /tmp/capstone/corpus-sil/qemu-arm2.sh silicon-mruby-bounds > $O/qemu.log 2>&1; rc=$?
    grep -q 'Another Capstone VM owns' /tmp/capstone/corpus-sil/vm-silicon-mruby-bounds-c81.up.log 2>/dev/null && [ $(date +%s) -lt $end ] && { sleep 60; continue; }
    break
  done
  echo "qemu rc=$rc $(tail -2 $O/qemu.log | tr '\n' ' ')"
  echo TWIN_MRUBY_QEMU_DONE
}
main "$@"; exit
