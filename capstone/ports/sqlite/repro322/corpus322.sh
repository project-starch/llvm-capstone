#!/usr/bin/env bash
# Control-arm corpus driver for the SQLite 3.22.0 temporal-bug study.
#   build <group> : compile every case into its share dir (cached sqlite3.o)
#   run   <group> : boot QEMU, run each .dom; resume across boots; isolate faulters
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)   # ports/sqlite/repro322
PORTS_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)                        # ports/sqlite
cd "$PORTS_DIR"
# Overridable so two checkouts (or two users) do not collide in /tmp.
export SQLITE322_TMP_ROOT=${SQLITE322_TMP_ROOT:-/tmp/capstone-322-repro}
source "$PORTS_DIR/../../tests/capstone-test-env.sh" 2>/dev/null

GROUP=${2:-core}
# CORPUS_SUBLET=1 builds the SECOND arm: identical in every respect except that the
# allocator carries the Sublet discipline (sublet/sublet-3220000-memsys5.patch). The two
# arms therefore differ by the allocator patch ALONE -- unlike Capstone-vs-CheriBSD, where
# ISA and OS change together. Its artefacts live in their own root so one arm never
# overwrites the other, and the sqlite3.o cache is off because the cached object is the
# unpatched one.
SUBLET=${CORPUS_SUBLET:-0}
if [ "$SUBLET" = 1 ]; then ROOT=$SQLITE322_TMP_ROOT/corpus-$GROUP-sublet
else ROOT=$SQLITE322_TMP_ROOT/corpus-$GROUP; fi
SHARE=$ROOT/share
OBJ=$ROOT/obj

# Under Sublet the pool is NOT the static sqlite_heap[]: it arrives from the host as a
# linear region (`h.user --arena N`), and memsys5 keeps aCtrl/aLink/aCap/aPar out of band
# in a second region (`--tables M`). The sizing is memsys5Init's under the port, for
# atoms = N/64: aCtrl one byte an atom, aLink 8, aCap 16, aPar 16 per possibly-split block.
sublet_tables() {   # $1 = arena bytes -> tables bytes
  local atoms=$(( $1 / 64 ))
  echo $(( ((atoms + 15) & ~15) + atoms*8 + atoms*16 + (atoms + 32)*16 + 64 ))
}
# A case may override the arena with -DSQLITE_HEAP_SIZE in its manifest `extra` column;
# the host region must then be the SAME size, or memsys5Init sizes its tables for one pool
# and carves another.
heap_of() { case "$*" in *-DSQLITE_HEAP_SIZE=*) echo "$*" | sed -E 's/.*-DSQLITE_HEAP_SIZE=([0-9]+).*/\1/' ;; *) echo 262144 ;; esac; }
MATHINC="-include $SCRIPT_DIR/repro322_math_decl.h"
case "$GROUP" in
  core) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION" ;;
  coreT) CF="-USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_TEMPDB" ;;
  fts5) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 $MATHINC" ;;
  fts5S) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 -USQLITE_OMIT_SHARED_CACHE $MATHINC" ;;
  fts3) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 $MATHINC" ;;
  ext) CF="-USQLITE_OMIT_INCRBLOB $MATHINC" ;;
  json) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_JSON1" ;;
  # rtree needs floating point, but this port builds with -DSQLITE_OMIT_FLOATING_POINT=1,
  # which does `#define double sqlite_int64` in sqliteInt.h AFTER sqlite3.h has already
  # typedef'd sqlite3_rtree_dbl as a real double -- so RtreeDValue and sqlite3_rtree_dbl
  # disagree and rtree.c will not compile.  -DSQLITE_RTREE_INT_ONLY makes BOTH typedefs
  # sqlite3_int64 and they agree again.  Cell layout is unchanged (RtreeValue int vs
  # float are both 4 bytes), and the host ASan oracle confirms both rtree cases still
  # reproduce under INT_ONLY at the same row counts.
  # row 6's bug is in fts3EvalNextRow()'s NESTED-OR branch, and nested query syntax
  # exists only with -DSQLITE_ENABLE_FTS3_PARENTHESIS. Without it the parentheses are
  # ordinary characters, a flat query runs, and the case returns 0 rows -- a PASS that
  # establishes nothing. Kept as its own group so the other fts3 cases keep building
  # against stock fts3 flags.
  fts3P) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS3_PARENTHESIS $MATHINC" ;;
  # The backported corrupt-database images. They need a FILESYSTEM inside the domain, not
  # just a database: the pager creates a rollback journal for any write, and ext/misc/memvfs.c
  # serves the main database only -- and takes its buffer as an integer in the URI, which on a
  # capability machine is a pointer forged from a scalar and faults on first use. Hence
  # repro322_memfs.c. Split by extension because fts3 and fts5 are separate builds.
  memfs3) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 $MATHINC" ;;
  memfs5) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 $MATHINC" ;;
  # The R2 fuzz-diff images, triggers re-derived 2026-10-03. Four of the five
  # are driven by `SELECT * FROM dbstat`, which does not exist in the binary
  # unless DBSTAT_VTAB is on -- so without this group the case would report
  # "no such table: dbstat" and score a clean pass having tested nothing.
  memfsp)  CF="-USQLITE_OMIT_INCRBLOB $MATHINC" ;;
  memfsdb) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_DBSTAT_VTAB $MATHINC" ;;
  rtree) CF="-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_RTREE -DSQLITE_RTREE_INT_ONLY -USQLITE_OMIT_SHARED_CACHE $MATHINC" ;;
  *) echo "unknown group $GROUP" >&2; exit 2 ;;
esac

manifest() {
  case "$GROUP" in
   core) cat <<EOF
case_mem5design.c   mem5design
case_agginfo.c      agginfo
case_backupattach.c backupattach
case_blobwrite.c    blobwrite
case_writable_schema.c wschema
case_mem5tagmap.c   mem5tag
case_lookaside_tagmap.c latag
case_lookaside_uaf.c    lauaf
case_fz09_searchwith.c  fz09
case_fz08_cursormoveto.c fz08
../sqlite_blobclose_domain.c blobclose
EOF
   ;;
   coreT) cat <<EOF
case_detach_trigger.c detachtrig
EOF
   ;;
   fts5S) cat <<EOF
case_fts5rank.c     fts5rank
case_fz12_fts5vocab_recursion.c fz12
EOF
   ;;
   ext) cat <<EOF
case_expert_rem.c   expertrem   -DSQLITE_HEAP_SIZE=1048576
case_spellfix_oom.c spellfixoom
EOF
   ;;
   json) cat <<EOF
case_json_each_static.c jsoneachstatic
case_json_each_root.c   jsoneachroot
case_jsondiag.c         jsondiag
EOF
   ;;
   fts3P) cat <<EOF
case_fts3p_probe.c     fts3pprobe
case_fts3_snippet_or.c fts3snipor
EOF
   ;;
   memfs3) cat <<EOF
case_2c7a73eaea_0.c  2c7a73eaea_0
case_a783931794_0.c  a783931794_0
case_c7def600bd_0.c  c7def600bd_0
EOF
   ;;
   memfsp) cat <<EOF
case_fz06_r2.c  fz06_r2
EOF
;;
   memfsdb) cat <<EOF
case_fz02_r2.c  fz02_r2
case_fz10_r2.c  fz10_r2
case_fz11_r2.c  fz11_r2
case_fz13_r2.c  fz13_r2
EOF
;;
   memfs5) cat <<EOF
case_415540ddaa_0.c  415540ddaa_0
case_8f5b14a5c2_0.c  8f5b14a5c2_0
case_bfe33f80dd_0.c  bfe33f80dd_0
case_33cf194218_0.c  33cf194218_0
case_634ac14488_0.c  634ac14488_0
case_adfb203a7d_0.c  adfb203a7d_0
case_174c21ff06_0.c  174c21ff06_0
EOF
   ;;
   rtree) cat <<EOF
case_rtree_probe.c    rtreeprobe
case_rtree_cursor.c   rtreecursor
case_rtree_inode0.c   rtreeinode0
case_rtree_static_bind.c rtreestatic
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
case_fts3_zterm.c      fts3zterm
case_fts3_offsets.c    fts3offsets
case_fts3_snippet.c    fts3snip
case_fts3_destroy_oom.c fts3destroyoom
case_fts3_static_bind.c staticbind
EOF
   ;;
  esac
}

# The extension sources the ext group compiles as extra TUs.
# The ext group compiles extension sources straight out of the 3.22.0 source tree.
# Unpack sqlite-src-3220000.zip (sha256 7bc5a3ce…) and point EXT_SRC_DIR at its ext/.
EXT_SRC_DIR=${EXT_SRC_DIR:-/home/zephyr/arms/sqlite/shared/sqlite-3.22.0-full/ext}

do_build() {
  mkdir -p "$SHARE" "$OBJ"
  local BASE_SRC=""
  case "$GROUP" in
    fts5|fts5S|fts3|fts3P|rtree) BASE_SRC="$SCRIPT_DIR/repro322_fts_stubs.c" ;;
    memfs3|memfs5|memfsp|memfsdb) BASE_SRC="$SCRIPT_DIR/repro322_fts_stubs.c $SCRIPT_DIR/repro322_memfs.c" ;;
  esac
  while read -r file tag extra; do
    [ -z "${file:-}" ] && continue
    # ext cases each pull in their own extension TU (and its include/defines)
    local EXTRA_SRC="$BASE_SRC" EXTRA_INC=""
    if [ "$GROUP" = ext ]; then
      case "$tag" in
        expertrem)   EXTRA_SRC="$EXT_SRC_DIR/expert/sqlite3expert.c"
                     EXTRA_INC="-I$EXT_SRC_DIR/expert" ;;
        # spellfix.c calls qsort; repro322_fts_stubs.c has the tag-preserving one.
        # -DSQLITE_CORE makes SQLITE_EXTENSION_INIT2 a no-op so the init is called directly.
        spellfixoom) EXTRA_SRC="$EXT_SRC_DIR/misc/spellfix.c $SCRIPT_DIR/repro322_fts_stubs.c"
                     EXTRA_INC="-DSQLITE_CORE -I$EXT_SRC_DIR/misc" ;;
      esac
    fi
    # The legacy standalone domain has its own domain_main and never captures the grant,
    # so under Sublet memsys5Init reads an empty slot and faults inside capstone_cap_base
    # before the case runs. That is the harness missing, not a result -- skip it here
    # rather than record a fault the arm did not earn.
    if [ "$SUBLET" = 1 ] && [ "$file" = "../sqlite_blobclose_domain.c" ]; then
      echo "== skip $tag (no Sublet grant glue in the standalone domain) =="; continue
    fi
    local SUBLET_ENV=() CACHE=1
    if [ "$SUBLET" = 1 ]; then
      CACHE=0
      SUBLET_ENV=(SQLITE_SUBLET_PATCH="$PORTS_DIR/sublet/sublet-3220000-memsys5.patch")
      extra="${extra:-} -DREPRO322_SUBLET=1"
    fi
    echo "== build $tag ($file) =="
    if env "${SUBLET_ENV[@]}" CORPUS_CACHE_SQLITE=$CACHE \
       OUT_DIR="$OBJ" OBJ_DIR="$OBJ" OUT_DOM="$SHARE/$tag.dom" \
       DOMAIN_SRC="$SCRIPT_DIR/$file" \
       DOMAIN_OPT_LEVEL="-O0" SQLITE_OPT_LEVEL="-O0" \
       DOMAIN_EXTRA_FLAGS="-DSQLITE_HEAP_SIZE=262144 $EXTRA_INC ${extra:-}" \
       DOMAIN_EXTRA_SRC="$EXTRA_SRC" \
       SQLITE_CFLAGS_EXTRA="$CF" \
       bash "$PORTS_DIR/build-sqlite-row322.sh" > "$SHARE/build-$tag.log" 2>&1
    then echo "  built $tag.dom"; else echo "  BUILD FAILED ($tag)"; grep -iE "error:" "$SHARE/build-$tag.log" | head -3; fi
  done < <(manifest)
  OUT_DIR="$SHARE" OUT_HOST="$SHARE/h.user" bash "$PORTS_DIR/build-sqlite-host.sh" >/dev/null 2>&1 && echo "== host built =="
}

# A fault pc is only evidence once it has a function name on it: `cause = 24` says the
# domain stopped, not what stopped it, and the same cause number covers "Sublet revoked the
# block the case then read" and "the harness handed memsys5 an empty slot". The domain is
# loaded at a base the log prints as pc_cap's bounds, so resolve against THAT and not a
# constant -- the base is not the same for every image.
where_fault() {
  local t=$1 log pc base off
  log=$(grep -la "capability fault" "$SHARE"/run-boot*.log 2>/dev/null | tail -1)
  [ -n "$log" ] || { echo "(capability fault; no log)"; return; }
  pc=$(grep -a -m1 -oE "domain halted by capability fault: cause = [0-9]+, pc = 0x[0-9a-f]+" "$log" | grep -oE "0x[0-9a-f]+$")
  base=$(grep -a -m1 -oE "pc_cap = C\([0-9a-f]+ \[[0-9a-f]+," "$log" | grep -oE "\[[0-9a-f]+," | tr -d "[,")
  [ -n "$pc" ] && [ -n "$base" ] || { echo "(capability fault; pc unresolved)"; return; }
  off=$(printf "0x%x" $(( pc - 0x$base + 0x10000 )))
  echo "(cause-24 at $(python3 "$SCRIPT_DIR/symf.py" "$SHARE/$t.dom" "$off" 2>/dev/null || echo "$off"), pc=$pc)"
}

do_run() {
  # The QEMU runner needs pexpect from the qemu-deps env. `set -uo pipefail` has no -e,
  # so a missing conda.sh used to be skipped silently and surface much later as an
  # unrelated pexpect failure. Fail here instead, with the reason.
  CONDA_SH=${CONDA_SH:-/home/miniconda/miniconda3/etc/profile.d/conda.sh}
  if [ ! -r "$CONDA_SH" ]; then
    echo "ERROR: cannot read $CONDA_SH; set CONDA_SH, or activate an env with pexpect yourself" >&2
    return 2
  fi
  # shellcheck disable=SC1090
  source "$CONDA_SH" && conda activate qemu-deps || {
    echo "ERROR: 'conda activate qemu-deps' failed; the QEMU runner needs pexpect" >&2
    return 2
  }
  local all=() ; declare -A ARENA_OF=()
  while read -r file tag extra; do
    [ -z "${file:-}" ] && continue
    [ -f "$SHARE/$tag.dom" ] || continue
    all+=("$tag"); ARENA_OF[$tag]=$(heap_of "${extra:-}")
  done < <(manifest)
  local pass="$SHARE/passed.txt" fault="$SHARE/faulted.txt" err="$SHARE/erred.txt"
  local infra="$SHARE/infra.txt"
  : > "$pass"; : > "$fault"; : > "$err"; : > "$infra"
  local boot=0
  while :; do
    local rem=()
    for t in "${all[@]}"; do grep -qxF "$t" "$pass" || grep -qxF "$t" "$fault" || grep -qxF "$t" "$err" || grep -qxF "$t" "$infra" || rem+=("$t"); done
    [ ${#rem[@]} -eq 0 ] && break
    boot=$((boot+1)); local log="$SHARE/run-boot$boot.log"
    echo "=== boot $boot: running ${rem[*]} ==="
    local gc="cp /mnt/host/h.user /tmp/h && chmod 0755 /tmp/h"
    for d in "${rem[@]}"; do
      local args=""
      if [ "$SUBLET" = 1 ]; then
        local a=${ARENA_OF[$d]:-262144}
        # --tail prints the payload WHILE the domain runs, so the markers of a domain that
        # later faults are not lost with it -- which is what makes a faulting Sublet run
        # interpretable at all (out_text is otherwise read only after the domain returns).
        args=" --tail --arena $a --tables $(sublet_tables "$a")"
      fi
      gc="$gc && (echo ==RUN $d==; /tmp/h /mnt/host/$d.dom$args; echo ==RC $d=\$?==)"
    done
    gc="$gc && echo ==ALLDONE=="
    local rc=1
    for attempt in 1 2 3 4; do
      # Boot-login fails FAST; the workload keeps the full multiplier budget.
      # Without this the login wait inherits 120 * 8 = 960 s, and a boot that is
      # already dead at 40 s costs 16 minutes before the retry. Measured: 5 of 17
      # attempts flaked, ~80 min of pure timeout against ~8 min of real work.
      CAPSTONE_QEMU_LOGIN_TIMEOUT="${CORPUS_LOGIN_TIMEOUT:-180}" \
      python3 "$PORTS_DIR/../../tests/runtime-qemu/run-domain-smoke.py" \
        --share-dir "$SHARE" --log-file "$log" --timeout-multiplier "${CORPUS_TIMEOUT_MULT:-8}" \
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
    # qemu died mid-batch. Only call it a FAULT if the log actually carries a capability
    # fault line. Without this check a plain boot stall or a prompt timeout is blamed on
    # rem[0], and in a SINGLE-CASE group that means ANY boot failure is recorded as a false
    # FAULT -- which is exactly how fts5rank was briefly mis-recorded. Infra failures go to
    # their own bucket so they can never be mistaken for a result.
    if [ $progressed -eq 0 ]; then
      local victim="${rem[0]}"
      if grep -aq "capability fault" "$log"; then
        echo "$victim" >> "$fault"; echo "  isolated faulter: $victim (rc=$rc)"
      else
        echo "$victim" >> "$infra"
        echo "  INFRA: no capability-fault line in $log; recording $victim as INFRA, not FAULT (rc=$rc)"
      fi
    fi
  done
  echo "===== RESULTS ($GROUP$( [ "$SUBLET" = 1 ] && echo " -- SUBLET arm" )) ====="
  for t in "${all[@]}"; do
    if grep -qxF "$t" "$pass"; then echo "PASS   $t (control ran to NOTRAP)";
    elif grep -qxF "$t" "$fault"; then echo "FAULT  $t $(where_fault "$t")";
    elif grep -qxF "$t" "$err"; then echo "ERR    $t (ran, no NOTRAP -- soft error path; see run-boot*.log)";
    elif grep -qxF "$t" "$infra"; then echo "INFRA  $t (boot stalled/timed out, NO capability fault -- not a result; re-run, consider a larger CORPUS_TIMEOUT_MULT)";
    else echo "NORUN  $t"; fi
  done
}

case "${1:-}" in
  build) do_build ;;
  run)   do_run ;;
  both)  do_build && do_run ;;
  *) echo "usage: corpus322.sh {build|run|both} <group>"; exit 2 ;;
esac
