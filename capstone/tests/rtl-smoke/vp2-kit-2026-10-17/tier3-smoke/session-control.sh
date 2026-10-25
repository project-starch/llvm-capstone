#!/bin/bash
# The appvm batch-session mechanism's own control, on QEMU: wireshark/wmem-repros virtual-malloc through
# tools/board/appvm-session.py on the v6-fix kit (collect -> one QEMU boot -> read), compared by image with the direct
# appvm run of the same builds on the same kit (tier3-fix/wmem/out-virtual-malloc). Usage: session-control.sh <scratch>.
main() {
  set -u
  local R=${1:?scratch dir} G=/tmp/capstone/wt-gate T=/tmp/capstone/vp2-prep/tier3-fix/wmem K=/tmp/capstone/vp2-prep/v6-fix/out/kit
  local RUN=$G/capstone/bug-corpora/wireshark/wmem-repros/shared/run-defects.py S=$R/session
  local ARGS=(--arm virtual-malloc --hosted-build $T/cases-OFF --controls-hosted-build $T/controls-OFF --llvm-bin /tmp/capstone/vp2-prep/llvm-bin-v6)
  [ -e "$R" ] && { echo "STOP: $R exists"; return 2; }
  mkdir -p "$R"
  python3 $G/capstone/bug-corpora/tools/board/appvm-session.py init $S --virtual-kit $K || return 1
  python3 $RUN $R/out-collect --state $S/state "${ARGS[@]}" --raw $R/raw-collect > $R/collect.log 2>&1; echo "collect rc=$? calls=$(wc -l < $S/plan.jsonl)"
  python3 $G/capstone/bug-corpora/tools/board/appvm-session.py boot $S > $R/boot.log 2>&1; echo "boot rc=$? :: $(tail -1 $R/boot.log)"
  python3 $RUN $R/out-read --state $S/state "${ARGS[@]}" --raw $R/raw-read > $R/read.log 2>&1; echo "read rc=$? :: $(tail -1 $R/read.log)"
  python3 $G/capstone/tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/compare-pilot.py $T/out-virtual-malloc $R/out-read; echo "compare rc=$?"
}
main "$@"; exit $?
