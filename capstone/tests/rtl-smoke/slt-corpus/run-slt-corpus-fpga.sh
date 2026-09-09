#!/usr/bin/env bash
# SQLLogicTest corpus on silicon: one SQLite domain per boot (the pool allows exactly one SLT domain
# per boot), the k800 control FIRST in every boot, cheap files first and the untested 4 MiB region
# last. Per boot: wait for the console, stage EXACTLY {lpc, k800.dom, host, dom, test} in BOTH
# buildroot dirs (prune only known campaign artifacts -- never package files such as sbi.dom; the
# preflight's C15 gate scans the whole overlay, so one boot's files at a time), rebuild the firmware,
# re-probe, run the stage driver, read the result from the run's OWN transcript segment (the console
# replays the previous boot on connect), compare to native, append a row draft.
#
#   usage: run-slt-corpus-fpga.sh <images-dir> <native.txt> <results-dir> [boots]
#   boots: space-separated name:host:dom:test:budget entries (default: the whole corpus)
#
# Results dir must be DURABLE (not /tmp). Every boot writes <name>.driver.log, <name>-raw.txt,
# <name>.result (silicon line, native line, verdict). A boot whose driver dies at connect() before
# any board action is retried; one that dies mid-boot is recorded VOID and left for a rerun.
set -u
IMG=${1:?images dir}; NATIVE=${2:?native tsv}; RES=${3:?results dir}; shift 3
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
BR=$ROOT/capstone/caplifive-system/sw/buildroot; O=$BR/overlay/test-domains; T=$BR/build/target/test-domains
FW=$BR/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin
mkdir -p "$RES"; LOG="$RES/campaign.log"
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
ARTIFACTS="sqslt1m.dom sqslt3.dom sqslt4.dom sqslt5.dom sqlite_host_1m.user sqlite_host_2m.user sqlite_host_4m.user sqlite_host_2m5.user sqm0o1.dom sqlite_host.user q_two.test negctl.test aggfunc.test select1.test select2.test select3.test select4.test select5.test"
DEFAULT_BOOTS="negctl:sqlite_host_1m.user:sqslt1m.dom:negctl.test:900 aggfunc:sqlite_host_1m.user:sqslt1m.dom:aggfunc.test:900 select2:sqlite_host_1m.user:sqslt1m.dom:select2.test:2400 select5:sqlite_host_2m5.user:sqslt5.dom:select5.test:3000 select3:sqlite_host_2m.user:sqslt3.dom:select3.test:4800 select4:sqlite_host_4m.user:sqslt4.dom:select4.test:5400"
BOOTS=${*:-$DEFAULT_BOOTS}
CORPUS=$(bash "$ROOT/capstone/benchmarks/sqlite/fetch-sqllogictest.sh" 2>/dev/null | tail -1)
export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"; export FPGA_FW=$FW
export FPGA_BITSTREAM=${FPGA_BITSTREAM:-caplifive_s12fix_5097eb166.bit} FPGA_BITSTREAM_UNVERIFIED=1 PREFLIGHT_ALLOW_SHORT=1
export ENTRY_STALL_S=420 EARLY_HALT_CONTROL=0 WEDGE_TRACER=0 HALT_MUX_READS=0
TOOL="CodeGen.so $(sha256sum "$ROOT/llvm/cmake-build-debug/lib/libLLVMCapstoneCodeGen.so" | cut -c1-16) $(stat -c %y "$ROOT/llvm/cmake-build-debug/lib/libLLVMCapstoneCodeGen.so" | cut -c1-16)"
say "CAMPAIGN START; images $IMG; toolchain $TOOL; boots: $BOOTS"
probe(){ curl -sS -m 10 -o /dev/null -w '%{http_code}' "$FPGA_URL" 2>/dev/null; }
testfile(){ [ -f "$IMG/../$1" ] && { echo "$IMG/../$1"; return; }; case "$1" in negctl.test) echo "$ROOT/capstone/benchmarks/sqlite/slt/negative-control.test";; aggfunc.test) echo "$CORPUS/evidence/slt_lang_aggfunc.test";; q_two.test) echo "$ROOT/capstone/benchmarks/sqlite/slt/q_two.test";; *) echo "$CORPUS/$1";; esac; }
for b in $BOOTS; do
  IFS=: read -r name host dom tf budget <<< "$b"
  [ -f "$IMG/$dom" ] && [ -f "$IMG/$host" ] || { say "SKIP $name: $dom / $host not in $IMG"; continue; }
  src=$(testfile "$tf"); [ -f "$src" ] || { say "SKIP $name: no test file $src"; continue; }
  for i in $(seq 1 180); do code=$(probe); [ "$code" != "000" ] && break; [ $((i % 5)) -eq 1 ] && say "console down before $name (probe $i)"; sleep 60; done
  [ "$code" != "000" ] || { say "!! console never answered in 3 h; stopping before $name"; break; }
  for d in "$O" "$T"; do for a in $ARTIFACTS; do case "$a" in "$host"|"$dom"|"$tf") ;; *) rm -f "$d/$a" ;; esac; done
    cp -f "$IMG/lpc" "$IMG/k800.dom" "$IMG/$host" "$IMG/$dom" "$d/"; cp -f "$src" "$d/$tf"; done
  say "=== $name: staged $(ls "$O" | tr '\n' ' ') | dom $(sha256sum "$IMG/$dom" | cut -c1-16) host $(sha256sum "$IMG/$host" | cut -c1-16)"
  cd "$BR"; FW0=$(sha256sum "$FW" | cut -c1-12); CC=$(realpath ../../../capstone-c)
  make build LINUX_PAYLOAD=1 A=linux-rebuild CAPSTONE_CC_PATH="$CC" > "$RES/bake-$name-linux.log" 2>&1 || { say "!! linux-rebuild FAILED for $name"; break; }
  make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$CC" > "$RES/bake-$name-sbi.log" 2>&1 || { say "!! opensbi-rebuild FAILED for $name"; break; }
  FW1=$(sha256sum "$FW" | cut -c1-12); say "    fw_payload $FW0 -> $FW1 ($(stat -c%s "$FW") B)"
  cd "$ROOT/capstone/tests/rtl-smoke"
  export SQLITE_HOST=/test-domains/$host SQLITE_STAGE_DOMS="/test-domains/lpc|k800:/test-domains/k800.dom,/test-domains/$dom:--slt /test-domains/$tf"
  export SQLITE_STAGE_TIMEOUT=$budget SQLITE_IDLE_S=$budget PROBE_SCOPED_OUT="$RES/$name.txt" PROBE_RAW_OUT="$RES/$name-raw.txt"
  for attempt in $(seq 1 12); do
    for i in $(seq 1 180); do code=$(probe); [ "$code" != "000" ] && break; [ $((i % 5)) -eq 1 ] && say "console down before $name (pre-driver probe $i)"; sleep 60; done
    say "=== BOOT $name (attempt $attempt): $SQLITE_STAGE_DOMS budget=$budget"
    timeout $((budget + 1500)) python3 -m fpga_driver.run_sqlite_stages_fpga > "$RES/$name.driver.log" 2>&1; rc=$?
    if grep -aq 'ConnectionError' "$RES/$name.driver.log" && ! grep -aq 'load_image\|power' "$RES/$name.driver.log"; then say "    connect() failed before any board action -- window closed; retrying $name"; sleep 60; continue; fi
    break
  done
  python3 "$ROOT/capstone/tests/rtl-smoke/slt-corpus/slt-read.py" "$RES" "$name" "$tf" "$NATIVE" > "$RES/$name.result" 2>&1
  say "--- $name driver rc=$rc fw=$FW1 toolchain=$TOOL"; sed 's/^/    /' "$RES/$name.result" | tee -a "$LOG"
  grep -aq 'preflight: BLOCKED\|HARD STOP' "$RES/$name.driver.log" && { say "!! preflight/hard stop -- stopping: $(grep -a 'BLOCK\|HARD STOP' "$RES/$name.driver.log" | head -3 | cut -c1-160 | tr '\n' ' ')"; break; }
done
say "CAMPAIGN DONE"; echo done > "$RES/campaign.marker"
