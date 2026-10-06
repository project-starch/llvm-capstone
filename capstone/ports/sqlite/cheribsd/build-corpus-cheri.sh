#!/bin/bash
# Cross-compile the SQLite 3.22.0 temporal-bug corpus for CheriBSD riscv64-purecap.
#
# Flag policy: the SQLite build flags are copied VERBATIM from the Capstone arm
# (ports/sqlite/build-sqlite-row322.sh + repro322/corpus322.sh) so that the only
# variable between the two arms is the ISA and OS, never which code paths exist.
# The single deliberate change is -DSQLITE_OS_OTHER=1 -> -DSQLITE_OS_UNIX=1,
# because CheriBSD has a real VFS whereas the Capstone domain is freestanding.
# Dropped, all freestanding-only: -ffreestanding, -fno-builtin, the capstone libc
# shim headers, the stub VFS, repro322_math_decl.h and repro322_fts_stubs.c
# (CheriBSD's libc supplies libm and a tag-safe qsort).
set -euo pipefail

# Paths. C is this directory, inside the repository, and holds the sources:
# cases/, poscontrol.c, repro322_common.h and the sibling scripts. WORK is
# where build output and run logs go, which must NOT be in the repository; it
# defaults to the out-of-tree directory these scripts were developed in, so
# behaviour is unchanged unless it is set. The toolchain and the pinned SQLite
# amalgamation are machine-specific and are overridable for the same reason.
C=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
WORK=${SQLITE_CHERI_WORK:-$HOME/arms/sqlite/cheribsd}
mkdir -p "$WORK"
SDK=${CHERI_SDK:-/home/zephyr/cheriBSD/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-/home/zephyr/cheriBSD/cheri/output/rootfs-riscv64-purecap}
CLANG=$SDK/bin/clang
# The source belongs to no single arm: all three share one pinned 3.22.0
# amalgamation. It used to live only under this arm's directory while the
# host oracle compiled it from there -- a cross-arm dependency -- so it was
# moved to shared/.
SRC=${SQLITE_AMALGAMATION:-/home/zephyr/arms/sqlite/shared/amalgamation}
EXT=${SQLITE_EXT:-/home/zephyr/arms/sqlite/shared/sqlite-3.22.0-full/ext}

GROUP=${1:?usage: build-corpus-cheri.sh <group|all>}
# PROBE=1: build against the reachability-probed amalgamation. Separate out/obj
# trees, because the plain binaries are the ones the recorded results came from.
PROBE=${PROBE:-}
if [ -n "$PROBE" ]; then
  AMALG=sqlite3-probe.c; PROBE_FLAGS=(-DLB_PROBE_BUILD=1)
  OUT=$WORK/out-probe; OBJ=$WORK/obj-probe
else
  AMALG=sqlite3-cheri.c; PROBE_FLAGS=()
  OUT=$WORK/out; OBJ=$WORK/obj
fi
mkdir -p "$OUT" "$OBJ"

PURECAP=(--target=riscv64-unknown-freebsd --sysroot="$SYSROOT"
         -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax)

# SYSALLOC=1 builds the SAME cases against the SYSTEM allocator instead of memsys5.
# SQLITE_ZERO_MALLOC makes the default allocator a stub that always fails, so without
# it dropped there is no allocator at all unless CONFIG_HEAP is set -- which is why a
# runtime switch cannot do this and a separate build is required.
SYSALLOC=${SYSALLOC:-}
SUF=""
# PURECAP ALIGNMENT, and without this the whole _sys arm is SIGBUS.
# With no SQLITE_MALLOCSIZE, 3.22.0's mem1.c keeps the chunk size in an 8-byte
# prefix of its own:
#     p = SQLITE_MALLOC(nByte+8); if(p){ p[0]=nByte; p++; }
# so every pointer SQLite hands out is malloc's, PLUS 8. riscv64-purecap malloc
# returns 16-byte-aligned memory, 16n+8 is not 16-aligned, and storing a
# capability there raises SIGBUS/BUS_ADRALN on the first allocation that holds
# one. On x86 glibc this is harmless, which is why the host arm never showed it.
# Defining SQLITE_MALLOCSIZE takes the other branch, which returns malloc's own
# pointer and asks the system for the size.
# NOT via SQLITE_USE_MALLOC_H: that path does #include <malloc.h>, which FreeBSD
# does not have -- malloc_usable_size() lives in <malloc_np.h>.
SYSALLOC_FLAGS=()
if [ -n "$SYSALLOC" ]; then
  SUF="_sys"
  SYSALLOC_FLAGS=(-DSQLITE_MALLOCSIZE=malloc_usable_size -include malloc_np.h)
else
  ZM="-DSQLITE_ZERO_MALLOC=1"
fi
SQLITE_DEFINES=(
  -DNDEBUG
  -DSQLITE_OS_UNIX=1                 # was SQLITE_OS_OTHER=1 on Capstone
  -DSQLITE_THREADSAFE=0
  -DSQLITE_DEFAULT_MEMSTATUS=0
  -DSQLITE_TEMP_STORE=3
  -DSQLITE_OMIT_LOAD_EXTENSION=1
  -DSQLITE_OMIT_LOCALTIME=1
  -DSQLITE_OMIT_MMAP=1
  -DSQLITE_OMIT_WAL=1
  -DSQLITE_OMIT_SHARED_CACHE=1
  -DSQLITE_OMIT_TEMPDB=1
  -DSQLITE_OMIT_AUTOINIT=1
  -DSQLITE_OMIT_COMPILEOPTION_DIAGS=1
  # -DSQLITE_OMIT_FLOATING_POINT=1 is DELIBERATELY NOT SET on this arm.
  # On Capstone it is a platform workaround (that freestanding target has no FP), and it
  # works there only because the build uses stub headers. Here sqlite3.c includes the real
  # <math.h> for FTS3/FTS4/RTREE, and the flag makes sqliteInt.h '#define double
  # sqlite_int64', so every 'double' in math.h becomes a redefinition error. CheriBSD
  # riscv64-purecap has hardware FP, so dropping the workaround is the faithful choice.
  # Side effect to remember: OMIT_FLOATING_POINT also implied OMIT_TRACE on the Capstone
  # arm, so sqlite3_expanded_sql() is functional here and returns NULL there.
  # SQLITE_RTREE_INT_ONLY is NOT set here: the Capstone arm needs it only because
  # OMIT_FLOATING_POINT makes RtreeDValue and sqlite3_rtree_dbl disagree there. With
  # real FP, rtree builds with its normal double coordinates (handoff item 4).
  # still match the Capstone arm.
  -DSQLITE_OMIT_UTF16=1
  -DSQLITE_OMIT_INCRBLOB=1
  -DSQLITE_OMIT_GET_TABLE=1
  -DSQLITE_OMIT_DEPRECATED=1
  -DSQLITE_OMIT_EXPLAIN=1
  -DSQLITE_OMIT_FOREIGN_KEY=1
  -DSQLITE_OMIT_JSON=1
  -DSQLITE_DQS=0
  -DSQLITE_UNTESTABLE=1
  -DSQLITE_ENABLE_MEMSYS5=1
  ${ZM:-}
  -DSQLITE_DEFAULT_LOOKASIDE=0,0
  -DYYSTACKDEPTH=1000
)
# The Capstone arm defaults to SQLITE_FEATURE_SET=deployed, which makes its
# SQLITE_RESTORE list EMPTY, and corpus322.sh never overrides it. So this arm must
# not restore those omissions either. (Applying them also breaks the build outright:
# with COMPILEOPTION_DIAGS restored, the compile-option table stringifies
# SQLITE_DEFAULT_LOOKASIDE=0,0 and the comma splits the macro argument.)
SQLITE_RESTORE=()

group_cflags() {   # mirrors corpus322.sh
  case "$1" in
    core)  echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION" ;;
    coreT) echo "-USQLITE_OMIT_INCRBLOB -USQLITE_OMIT_TEMPDB" ;;
    fts5)  echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5" ;;
    fts5S) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5 -USQLITE_OMIT_SHARED_CACHE" ;;
    fts3)  echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4" ;;
    fts3P) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS3_PARENTHESIS" ;;
    lbcore) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_PREUPDATE_HOOK -DSQLITE_COUNTOFVIEW_OPTIMIZATION" ;;
    lbfts3) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4" ;;
    lbfts5) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS5" ;;
    # The R2 fuzz-diff images whose triggers were re-derived 2026-10-03.
    r2p)  echo "-USQLITE_OMIT_INCRBLOB" ;;
    r2db) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_DBSTAT_VTAB" ;;
    ext)   echo "-USQLITE_OMIT_INCRBLOB" ;;
    json)  echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_JSON1" ;;
    rtree) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_RTREE -USQLITE_OMIT_SHARED_CACHE" ;;
    probes) echo "-USQLITE_OMIT_INCRBLOB -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS3_PARENTHESIS -DSQLITE_ENABLE_FTS5 -DSQLITE_ENABLE_RTREE -DSQLITE_ENABLE_JSON1 -USQLITE_OMIT_SHARED_CACHE" ;;
    *) echo "unknown group: $1" >&2; exit 1 ;;
  esac
}

# tag -> source.
# EXCLUDED, all Capstone-only and none of them part of the 25-domain corpus tally:
#   case_mem5tagmap.c (mem5tag), case_lookaside_tagmap.c (latag),
#   case_lookaside_uaf.c (lauaf)  -- all use __builtin_capstone_cap_get_tag to inspect
#     capability tags directly; the CHERI equivalent is __builtin_cheri_tag_get, so they
#     could be ported later as diagnostics.
#   ../sqlite_blobclose_domain.c (blobclose, row 5) -- predates the group harness and
#     talks to the old sqlite_hostcall.h interface; needs a rewrite onto REPRO322_MAIN.
manifest() {
  # THE 19 ESTABLISHED BUGS (task-checkpoints/HANDOFF_19_BUGS_CHERIBSD.md).
  # Capstone split: 12 silent + 7 faulted.
  # Six domains were once removed from here -- agginfo, blobwrite, fts3offsets,
  # fts3snip, fts3zterm, fts5near -- on the argument that "no input reaches a memory
  # error, so they are not bugs of this corpus". RESTORED 2026-10-05. That argument
  # overstated the evidence: each has a real upstream fix commit and a live_proof that
  # is a line-numbered inspection of the 3.22.0 tree, and each runs on BOTH Capstone
  # arms from these same sources. What we actually have is a trigger that completes
  # without an observable memory error, which is a result and not a disqualification --
  # and without a defect-site probe we cannot even tell whether it reached the defect.
  # They belong in the denominator; a silent row on all three arms is data. The 5 baseline probes are in the "probes" group: diagnostics, never counted.
  case "$1" in
    r2p) cat <<M
case_fz06_r2.c  fz06_r2
M
;;
    r2db) cat <<M
case_fz02_r2.c  fz02_r2
case_fz10_r2.c  fz10_r2
case_fz11_r2.c  fz11_r2
case_fz13_r2.c  fz13_r2
M
;;
    lbcore) cat <<M
case_fz09_searchwith.c    fz09
case_fz08_cursormoveto.c  fz08
M
;;
    lbfts3) cat <<M
case_2c7a73eaea_0.c  2c7a73eaea_0  -DSQLITE_HEAP_SIZE=8388608
case_a783931794_0.c  a783931794_0
case_c7def600bd_0.c  c7def600bd_0
M
;;
    lbfts5) cat <<M
case_fz12_fts5vocab_recursion.c  fz12
case_415540ddaa_0.c  415540ddaa_0
case_8f5b14a5c2_0.c  8f5b14a5c2_0
case_bfe33f80dd_0.c  bfe33f80dd_0
case_33cf194218_0.c  33cf194218_0
case_634ac14488_0.c  634ac14488_0
case_adfb203a7d_0.c  adfb203a7d_0
case_174c21ff06_0.c  174c21ff06_0
M
;;
    core)  cat <<M
case_blobclose.c        blobclose
case_writable_schema.c  wschema
case_mem5design.c       mem5design
case_backupattach.c     backupattach
case_agginfo.c          agginfo
case_blobwrite.c        blobwrite
M
;;
    coreT) echo "case_detach_trigger.c detachtrig" ;;
    ext)   cat <<M
case_expert_rem.c   expertrem   -DSQLITE_HEAP_SIZE=1048576
case_spellfix_oom.c spellfixoom -static
M
;;
    fts5)  cat <<M
case_fts5vocab_eof.c   fts5vocabeof
case_fts5structwrite.c fts5structwrite
case_fts5inplace.c     fts5inplace
case_fts5near.c        fts5near
M
;;
    fts5S) echo "case_fts5rank.c fts5rank" ;;
    json)  cat <<M
case_json_each_static.c jsoneachstatic
case_json_each_root.c   jsoneachroot
M
;;
    fts3)  cat <<M
case_fts3_static_bind.c staticbind
case_fts3_destroy_oom.c fts3destroyoom
case_fts3_offsets.c     fts3offsets
case_fts3_snippet.c     fts3snip
case_fts3_zterm.c       fts3zterm
M
;;
    fts3P) echo "case_fts3_snippet_or.c fts3snipor" ;;
    rtree) cat <<M
case_rtree_static_bind.c rtreestatic
case_rtree_cursor.c      rtreecursor
case_rtree_inode0.c      rtreeinode0
M
;;
    probes) cat <<M
case_fts3probe.c    fts3probe
case_fts3p_probe.c  fts3pprobe
case_fts5probe.c    fts5probe
case_rtree_probe.c  rtreeprobe
case_jsondiag.c     jsondiag
M
;;
  esac
}

build_group() {
  local g="$1" cf; cf=$(group_cflags "$g")
  echo "=== group $g  [$cf] ==="
  # sqlite3.o once per group.
  # The cache key MUST include the compile flags, not just the group name.
  # It used to be "sqlite3-$g$SUF.o" alone, and when SYSALLOC_FLAGS gained the
  # purecap alignment fix (-DSQLITE_MALLOCSIZE) the stale object from before the
  # fix was silently relinked -- the binaries looked rebuilt and still had the bug.
  local key; key=$(printf %s "${SQLITE_DEFINES[*]} ${SQLITE_RESTORE[*]:-} ${SYSALLOC_FLAGS[*]:-} ${PROBE_FLAGS[*]:-} $AMALG $cf" | sha256sum | cut -c1-10)
  local so="$OBJ/sqlite3-$g$SUF-$key.o"
  if [ ! -f "$so" ]; then
    # shellcheck disable=SC2086
    "$CLANG" "${PURECAP[@]}" -O0 -I"$SRC" \
      "${SQLITE_DEFINES[@]}" "${SQLITE_RESTORE[@]}" "${SYSALLOC_FLAGS[@]:-}" ${PROBE_FLAGS[@]+"${PROBE_FLAGS[@]}"} $cf \
      -c "$SRC/$AMALG" -o "$so"
    echo "  sqlite3.o built"
  else
    echo "  sqlite3.o cached"
  fi
  local src tag extra
  while read -r src tag extra; do
    [ -z "${src:-}" ] && continue
    local xsrc=() xinc=()
    case "$tag" in
      spellfixoom) xsrc=("$SRC/spellfix.c"); xinc=(-DSQLITE_CORE -I"$EXT/misc") ;;
      expertrem)   if [ -n "$PROBE" ]; then xsrc=("$SRC/sqlite3expert-probe.c");
                   else xsrc=("$EXT/expert/sqlite3expert.c"); fi
                   xinc=(-DSQLITE_CORE -I"$EXT/expert") ;;
    esac
    # shellcheck disable=SC2086
    if "$CLANG" "${PURECAP[@]}" -O0 -I"$C" -I"$SRC" \
        "${SQLITE_DEFINES[@]}" "${SQLITE_RESTORE[@]}" "${SYSALLOC_FLAGS[@]:-}" ${PROBE_FLAGS[@]+"${PROBE_FLAGS[@]}"} $cf \
        -DSQLITE_HEAP_SIZE=262144 ${extra:-} "${xinc[@]:-}" \
        "$C/cases/$src" "${xsrc[@]:-}" "$so" -lm -o "$OUT/$tag$SUF" 2>"$OBJ/$tag.err"; then
      printf "  %-16s OK\n" "$tag"
    else
      printf "  %-16s FAIL (see obj/%s.err)\n" "$tag" "$tag"
      head -4 "$OBJ/$tag.err" | sed "s/^/      /"
    fi
  done < <(manifest "$g")
}

# positive control: plain system-malloc UAF, must trap under revocation
build_poscontrol() {
  "$CLANG" "${PURECAP[@]}" -O0 "$C/poscontrol.c" -o "$OUT/poscontrol"
  echo "  poscontrol       OK"
}

if [ "$GROUP" = all ]; then
  build_poscontrol
  # lbcore, lbfts3 and lbfts5 were missing from this list, and "all" is a
  # dangerous name for a list that omits groups: the 13 cases in them kept
  # whatever binary they last had while every other case was rebuilt, and
  # nothing in the output said so. That bit on 2026-10-05 -- a change to
  # repro322_common.h reached 30 of the 43 corpus cases, and the other 13 ran
  # from binaries dated 2026-10-03. The build loop recompiles unconditionally,
  # so there was no FAIL line and no stale-object warning to notice.
  # r2p and r2db stay out on purpose: those 5 _r2 images carry a host-ASan
  # signature but no derived trigger, so corpus.json does not count them as
  # cases.
  # r2p and r2db are in this list now. They were left out when the five R2
  # images were believed to be non-cases; the corpus note names fz01, fz03,
  # fz04, fz05 and fz07 as non-cases, and these groups hold fz02, fz06, fz10,
  # fz11 and fz13 -- four of which are cases 29 to 32. Building fz13_r2 too is
  # harmless: the runners take their case list from tags.tsv, which marks it
  # out of scope, so it is built and not run.
  for g in core coreT fts5 fts5S fts3 fts3P ext json rtree lbcore lbfts3 lbfts5 r2p r2db probes; do build_group "$g"; done
else
  build_group "$GROUP"
fi
