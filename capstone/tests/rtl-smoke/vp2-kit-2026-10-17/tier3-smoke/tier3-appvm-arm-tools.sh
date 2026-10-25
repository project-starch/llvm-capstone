#!/bin/bash
# tier3-appvm-arm.sh (11967b378bda73c8) with the session tool taken from /tmp/capstone/wt-tools, a clean detached worktree
# of dev 93ebde2f53b9, instead of wt-gate, which is held for the compiler lane's batch references; and the SDK checker from the same tree
# (wt-gate has none; it is byte-identical to the pinned trees'). Everything else is byte-identical. Header of the original:
# One appvm corpus arm through the batch session (tools/board/appvm-session.py): collect -> one boot -> read, on either
# kit, so the QEMU prediction and the board reading run the same mechanism and differ only in the kit.
# Usage: tier3-appvm-arm.sh --board-kit|--virtual-kit <kit> <out root> <runner> <runner args...>
#   the runner is invoked as: <runner> <out root>/out --state <session>/state <runner args...> --raw <out root>/raw-<phase>
#   (the wmem/mmgr run-defects.py form, output positional). Ends by itself; refuses an existing out root.
# EXPECT_IMAGES=<list>: after the collect, the collected images must equal that sha256 list (the prediction's), or the
# script stops before the boot.
# Before anything is created or booted, every build the runner arguments name (--build, --hosted-build, --controls-build,
# --controls-hosted-build, --prebuilt, --prebuilt-controls, --sdk; perl's --perl build root; mruby's --image and --capi-dir,
# whose SDK is the nearest ancestor holding one, up to eight levels, since mruby's image sits five levels below its build
# root's sdk/) must have been made with an effective -fno-builtin, or with a recorded builtins_allowed (C-81,
# bug-corpora/tools/check-sdk-builtins.py); otherwise, or when no build is named, it stops with exit 78.
sdk_up() {   # the nearest of <path>'s directory and its ancestors, up to eight levels, that holds an SDK (mruby's layout)
  local d=$1 i c
  [ -d "$d" ] || d=$(dirname "$d")
  for i in 0 1 2 3 4 5 6 7 8; do
    for c in sdk/sdk.json sdk.json image/sdk/sdk.json source/runtime/sdk.json runtime/sdk.json runtime-virtual/sdk/sdk.json; do
      [ -f "$d/$c" ] && { echo "$d/$(dirname "$c")"; return 0; }
    done
    [ "$d" = / ] && break; d=$(dirname "$d")
  done
  return 1
}
sdk_gate() {
  local chk targets=() extra=() prev= t
  chk=/tmp/capstone/wt-tools/capstone/bug-corpora/tools/check-sdk-builtins.py   # change 2: wt-gate has no checker; this one is byte-identical to wt-mp's and wt-ref2's (5abdb9ad)
  for w in "$@"; do
    case $prev in
      --build|--hosted-build|--controls-build|--controls-hosted-build|--prebuilt|--prebuilt-controls|--sdk|--perl) targets+=("$w");;
      --image|--capi-dir) t=$(sdk_up "$w") || { echo "STOP: SDK-BUILTINS no SDK above $prev $w (no data is a refusal)"; return 78; }
                          targets+=("$t");;
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
  local G=/tmp/capstone/wt-tools S=$R/session rc   # the ONE change from tier3-appvm-arm.sh 11967b37: the session tool's tree (dev 93ebde2f53b9: the board step bound)
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
