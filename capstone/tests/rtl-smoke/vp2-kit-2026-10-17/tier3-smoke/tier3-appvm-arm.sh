#!/bin/bash
# One appvm corpus arm through the batch session (tools/board/appvm-session.py): collect -> one boot -> read, on either
# kit, so the QEMU prediction and the board reading run the same mechanism and differ only in the kit.
# Usage: tier3-appvm-arm.sh --board-kit|--virtual-kit <kit> <out root> <runner> <runner args...>
#   the runner is invoked as: <runner> <out root>/out --state <session>/state <runner args...> --raw <out root>/raw-<phase>
#   (the wmem/mmgr run-defects.py form, output positional). Ends by itself; refuses an existing out root.
# EXPECT_IMAGES=<list>: after the collect, the collected images must equal that sha256 list (the prediction's), or the
# script stops before the boot.
# Before anything is created or booted, every build the runner arguments name (--build, --hosted-build, --controls-build,
# --controls-hosted-build, --prebuilt, --prebuilt-controls, --sdk) must have been made with an effective -fno-builtin, or
# with a recorded builtins_allowed (C-81, bug-corpora/tools/check-sdk-builtins.py); otherwise, or when no build is named,
# it stops with exit 78.
sdk_gate() {
  local chk targets=() extra=() prev=
  chk="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/bug-corpora/tools/check-sdk-builtins.py"
  for w in "$@"; do
    case $prev in
      --build|--hosted-build|--controls-build|--controls-hosted-build|--prebuilt|--prebuilt-controls|--sdk) targets+=("$w");;
      --cc-arg) extra+=(--extra "$w");;
    esac
    prev=$w
  done
  [ ${#targets[@]} -gt 0 ] || { echo "STOP: SDK-BUILTINS no build named in the runner arguments (no data is a refusal)"; return 78; }
  python3 "$chk" "${extra[@]}" "${targets[@]}" || { echo "STOP: SDK-BUILTINS refused (exit $?)"; return 78; }
}
main() {
  set -u
  local KK=${1:?--board-kit or --virtual-kit} K=${2:?kit} R=${3:?out root} RUN=${4:?runner}; shift 4
  local G=/tmp/capstone/wt-gate S=$R/session rc
  case $KK in --board-kit|--virtual-kit) ;; *) echo "STOP: $KK is neither --board-kit nor --virtual-kit"; return 2;; esac
  [ -e "$R" ] && { echo "STOP: $R exists"; return 2; }
  sdk_gate "$@" || return 78
  mkdir -p "$R"
  python3 $G/capstone/bug-corpora/tools/board/appvm-session.py init $S $KK $K > $R/init.log 2>&1 || { echo "STOP: init failed: $(tail -1 $R/init.log)"; return 1; }
  python3 $RUN $R/out-collect --state $S/state "$@" --raw $R/raw-collect > $R/collect.log 2>&1; rc=$?
  echo "APPVM collect rc=$rc calls=$(wc -l < $S/plan.jsonl 2>/dev/null) $(date +%T)"
  [ -s $S/plan.jsonl ] || { echo "STOP: nothing collected"; return 1; }
  if [ -n "${EXPECT_IMAGES:-}" ]; then
    diff <(cd $S/collect/images && sha256sum $(ls | sort)) "$EXPECT_IMAGES" > /dev/null \
      || { echo "STOP: the collected images differ from $EXPECT_IMAGES"; return 1; }
    echo "APPVM images equal $EXPECT_IMAGES ($(wc -l < "$EXPECT_IMAGES"))"
  fi
  python3 $G/capstone/bug-corpora/tools/board/appvm-session.py boot $S > $R/boot.log 2>&1; rc=$?
  echo "APPVM boot rc=$rc $(date +%T) :: $(tail -1 $R/boot.log)"
  python3 $RUN $R/out --state $S/state "$@" --raw $R/raw-read > $R/read.log 2>&1; rc=$?
  echo "APPVM read rc=$rc $(date +%T) :: $(grep -a '^--- ' $R/read.log | tail -1)"
}
main "$@"; exit $?
