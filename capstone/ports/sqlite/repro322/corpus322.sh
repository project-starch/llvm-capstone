#!/usr/bin/env bash
# Control-arm corpus driver for the SQLite 3.22.0 temporal-bug study.
#   build <group> : compile every case into its share dir (cached sqlite3.o)
#   run   <group> : boot QEMU, run each .dom; resume across boots; isolate faulters
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)   # ports/sqlite/repro322
PORTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)                        # ports/sqlite
cd "$PORTS_DIR"
export SQLITE322_TMP_ROOT=/tmp/capstone-322-repro
source "$PORTS_DIR/../../tests/capstone-test-env.sh" 2>/dev/null

GROUP=${2:-core}
ROOT=/tmp/capstone-322-repro/corpus-$GROUP
SHARE=$ROOT/share
OBJ=$ROOT/obj
MATHINC="-include $SCRIPT_DIR/repro322_math_decl.h"
case "$GROUP" in
  core) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION" ;;
  coreT) CF="-USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_TEMPDB" ;;
  fts5) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 $MATHINC" ;;
  fts5S) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 -USQLITE_OMIT_SHARED_CACHE $MATHINC" ;;
  fts3) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 $MATHINC" ;;
  *) echo "unknown group $GROUP" >&2; exit 2 ;;
esac

manifest() {
  case "$GROUP" in
   core) cat <<EOF
case_mem5design.c   mem5design
case_agginfo.c      agginfo
case_backupattach.c backupattach
case_blobwrite.c    blobwrite
EOF
   ;;
   coreT) cat <<EOF
case_detach_trigger.c detachtrig
EOF
   ;;
   fts5S) cat <<EOF
case_fts5rank.c        fts5rank
EOF
   ;;
   fts5) cat <<EOF
case_fts5probe.c fts5probe
case_fts5vocab_eof.c   fts5vocabeof
case_fts5structwrite.c fts5structwrite
case_fts5near.c        fts5near
case_fts5inplace.c     fts5inplace
EOF
   ;;
   fts3) cat <<EOF
case_fts3probe.c fts3probe
case_fts3_snippet_or.c fts3snipor
case_fts3_zterm.c      fts3zterm
case_fts3_offsets.c    fts3offsets
case_fts3_snippet.c    fts3snip
EOF
   ;;
  esac
}

do_build() {
  mkdir -p "$SHARE" "$OBJ"
  local EXTRA_SRC=""
  case "$GROUP" in fts5|fts5S|fts3) EXTRA_SRC="$SCRIPT_DIR/repro322_fts_stubs.c" ;; esac
  while read -r file tag extra; do
    [ -z "${file:-}" ] && continue
    echo "== build $tag ($file) =="
    if CORPUS_CACHE_SQLITE=1 \
       OUT_DIR="$OBJ" OBJ_DIR="$OBJ" OUT_DOM="$SHARE/$tag.dom" \
       DOMAIN_SRC="$SCRIPT_DIR/$file" \
       DOMAIN_OPT_LEVEL="-O0" SQLITE_OPT_LEVEL="-O0" \
       DOMAIN_EXTRA_FLAGS="-DSQLITE_HEAP_SIZE=262144 ${extra:-}" \
       DOMAIN_EXTRA_SRC="$EXTRA_SRC" \
       SQLITE_CFLAGS_EXTRA="$CF" \
       bash "$PORTS_DIR/build-sqlite-row322.sh" > "$SHARE/build-$tag.log" 2>&1
    then echo "  built $tag.dom"; else echo "  BUILD FAILED ($tag)"; grep -iE "error:" "$SHARE/build-$tag.log" | head -3; fi
  done < <(manifest)
  OUT_DIR="$SHARE" OUT_HOST="$SHARE/h.user" bash "$PORTS_DIR/build-sqlite-host.sh" >/dev/null 2>&1 && echo "== host built =="
}

do_run() {
  source /home/miniconda/miniconda3/etc/profile.d/conda.sh && conda activate qemu-deps
  local all=() ; while read -r file tag extra; do [ -z "${file:-}" ] && continue; [ -f "$SHARE/$tag.dom" ] && all+=("$tag"); done < <(manifest)
  local pass="$SHARE/passed.txt" fault="$SHARE/faulted.txt" err="$SHARE/erred.txt"
  : > "$pass"; : > "$fault"; : > "$err"
  local boot=0
  while :; do
    local rem=()
    for t in "${all[@]}"; do grep -qxF "$t" "$pass" || grep -qxF "$t" "$fault" || grep -qxF "$t" "$err" || rem+=("$t"); done
    [ ${#rem[@]} -eq 0 ] && break
    boot=$((boot+1)); local log="$SHARE/run-boot$boot.log"
    echo "=== boot $boot: running ${rem[*]} ==="
    local gc="cp /mnt/host/h.user /tmp/h && chmod 0755 /tmp/h"
    for d in "${rem[@]}"; do gc="$gc && (echo ==RUN $d==; /tmp/h /mnt/host/$d.dom; echo ==RC $d=\$?==)"; done
    gc="$gc && echo ==ALLDONE=="
    local rc=1
    for attempt in 1 2 3 4; do
      python3 "$PORTS_DIR/../../tests/runtime-qemu/run-domain-smoke.py" \
        --share-dir "$SHARE" --log-file "$log" --timeout-multiplier 8 \
        --guest-command "$gc" --success-marker "==ALLDONE==" > "$SHARE/boot$boot-attempt$attempt.log" 2>&1
      rc=$?
      [ $rc -eq 75 ] && { echo "  infra flake, retry"; continue; }
      break
    done
    local progressed=0
    for d in "${rem[@]}"; do
      if grep -aq "$d NOTRAP done" "$log" 2>/dev/null; then echo "$d" >> "$pass"; progressed=1; fi
    done
    if [ $rc -eq 75 ]; then echo "  INFRA: boot kept flaking; stopping"; break; fi
    if [ $rc -eq 0 ]; then
      # ALLDONE: every domain returned. Non-NOTRAP ones took a soft-error path.
      for d in "${rem[@]}"; do grep -qxF "$d" "$pass" || { echo "$d" >> "$err"; echo "  ran-no-NOTRAP: $d"; }; done
      continue
    fi
    # qemu died mid-batch (capability fault or crash)
    if [ $progressed -eq 0 ]; then
      local victim="${rem[0]}"; echo "$victim" >> "$fault"; echo "  isolated faulter: $victim (rc=$rc)"
    fi
  done
  echo "===== RESULTS ($GROUP) ====="
  for t in "${all[@]}"; do
    if grep -qxF "$t" "$pass"; then echo "PASS   $t (control ran to NOTRAP)";
    elif grep -qxF "$t" "$fault"; then echo "FAULT  $t (base-capstone capability fault; see run-boot*.log)";
    elif grep -qxF "$t" "$err"; then echo "ERR    $t (ran, no NOTRAP -- soft error path; see run-boot*.log)";
    else echo "NORUN  $t"; fi
  done
}

case "${1:-}" in
  build) do_build ;;
  run)   do_run ;;
  both)  do_build && do_run ;;
  *) echo "usage: corpus322.sh {build|run|both} <group>"; exit 2 ;;
esac
