#!/bin/bash
# Stubbed dry run of objb-first.sh: nothing reaches a board, the hub or the flash. The REAL objb-first.sh,
# make-objb-kit.sh (into a scratch kit root), chain-board.sh, spec-board.sh, run-stage.py, check-objb-boot.py and
# make-objb-stages.py --arm (on a scratch copy of the stages) run; these are stubbed:
#  OB_WAITF  -> a file that holds `TC DONE` (b4-t1c ended);   OB_BUSY / CB_BUSY / SB_BUSY -> true;
#  OB_STORE  -> store-stub.sh (rc 1 for an unknown name, 0 for bitstream 3);   OB_BOOK / SB_BOOK -> book-stub.sh;
#  OB_FLASH  -> flash-stub.sh (rc $OB_FLASH_RC, default 0);   OB_NVRB -> exits 3 (name matches, no sha256 exposed);
#  SB_RUN    -> sbrun-stub.sh: the REAL run-stage.py with BOARDVM_STUB_TRANSCRIPT = the stage's stub.
# Usage: run-dryrun.sh <case> <scratch dir>. Cases: pos, neg-sha, neg-nobit, neg-b4, neg-flash, neg-bare-stop,
# neg-bare-void, neg-bare-s3, neg-void, neg-control. The bare kit (objb-bare.sh) runs with supbare-stub.py as its runner (OBB_SCEN).
main() {
  set -u
  local CASE=${1:?case} T=${2:?scratch dir} D H
  D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); H=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}
  rm -rf "$T"; mkdir -p "$T/kits" "$T/stages"
  cp -a "$H/perf/objb/board/stages/." "$T/stages/"
  echo "TC DONE 20:00:00" > "$T/b4-t1c.log"
  printf 'not a bitstream, dry run\n' > "$T/src.bit"
  local NAME=caplifive_vp2c_603528b8b BIT=$T/bit/caplifive_vp2c_603528b8b.bit SHA
  SHA=$(sha256sum "$T/src.bit" | cut -d' ' -f1)
  mkdir -p "$T/bit"; [ "$CASE" = neg-nobit ] || cp "$T/src.bit" "$BIT"
  [ "$CASE" = neg-sha ] && SHA=$(printf '0%.0s' $(seq 64))
  [ "$CASE" = neg-b4 ] && echo "TC BOOT BEGIN" > "$T/b4-t1c.log"
  export OB_ROOT=$T/out OB_STAGES=$T/stages OB_WAITF=$T/b4-t1c.log OB_KDST=$T/kits OB_SLEEP=1 OB_SETTLE=0
  export OB_WAITMAX=3 OB_BITMAX=3 OB_BUSY=true OB_STORE="bash $D/store-stub.sh" OB_BOOK="bash $D/book-stub.sh"
  export OB_FLASH="bash $D/flash-stub.sh" OB_NVRB="bash $D/nvrb-stub.sh" OB_FLASH_RC=0
  [ "$CASE" = neg-flash ] && export OB_FLASH_RC=1
  export CB_BUSY=true SB_BUSY=true CB_POLL_S=1 SB_BOOK="bash $D/book-stub.sh" SB_RUN="bash $D/sbrun-stub.sh"
  export OB_STUB_DIR=$H/perf/objb/board/dryrun/stubs OB_PILOT_VARIANT=
  case $CASE in neg-void|neg-control) export OB_PILOT_VARIANT=$CASE;; esac
  # the bare identity kit: the REAL objb-bare.sh and grader; the runner, busy check and hub stubbed
  export OBB_RUN="python3 $D/supbare-stub.py" OBB_BUSY=true OBB_BOOK="bash $D/book-stub.sh" OBB_SCEN=objb-confirmed
  case $CASE in neg-bare-stop) export OBB_SCEN=objb-stop;; neg-bare-void) export OBB_SCEN=objb-void;; neg-bare-s3) export OBB_SCEN=objb-s3-minstret;; esac
  unset BOARDVM_STUB_TRANSCRIPT
  taskset -c 16-31,48-63 bash "$D/../objb-first.sh" "$T/objb.log" "$BIT" "$SHA"; local rc=$?
  echo "== $CASE: objb-first rc=$rc"
  grep -E '^OB |^  (objb|  C|OBJB-BARE)|CB STOP|SB STOP|END perf-objb' "$T/objb.log" | sed 's/^/   /'
  echo "   armed: $(ls "$T"/stages/*/plan.json 2>/dev/null | wc -l) disarmed: $(ls "$T"/stages/*/plan.json.disarmed 2>/dev/null | wc -l)"
  echo "   runner calls: $(wc -l < "$T/sbrun-argv.log" 2>/dev/null || echo 0)"
}
main "$@"; exit $?
