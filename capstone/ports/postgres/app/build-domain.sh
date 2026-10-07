#!/usr/bin/env bash
# PostgreSQL 17.5's backend as a Capstone musl domain: musl and the port runtime
# from a given llvm-capstone tree, the backend configured and compiled with the
# port's patches and settings, and one link. The output is $PG_SU_ROOT/link/postgres.dom
# and link/undefined.txt, the symbols the link could not resolve (empty on a full link).
#
#   RUNTIME_REPO=<llvm-capstone tree whose musl-capstone runtime to use> \
#     bash build-domain.sh [<postgresql-17.5.tar.bz2>]
#
# RUNTIME_REPO defaults to this tree. The runtime rows of the plan live on the
# runtime/* branches until they are in dev, so a build that needs them names a
# worktree of the newest one. Needs CAPSTONE_LLVM_BUILD_DIR (clang, ld.lld).
#
# Settings, and why (plan: docs/plans/postgres-single-user-port.md, layer 2):
#   CFLAGS -O1, no -g            Common release setting for the CheriBSD pair;
#                                  Assignment Tracking (C-50) runs only with debug info
#   -DWAIT_USE_POLL              the latch on poll + a self-pipe, not epoll + signalfd
#   pgac_cv_computed_goto=no     a table of &&labels loads untagged (the Lua/CPython trap)
#   MAXIMUM_ALIGNOF 16           configure derives it from long/double, not pointers;
#                                every palloc and tuple would put capabilities on 8-byte slots
#   patches/0001                 the executor's ExprEvalStep guard, for 16-byte pointers
#   patches/0002                 aset's smallest chunk holds a capability (the mmgr port's)
#   level0 arena                 PGSU_ARENA_BYTES, default 64 MiB: the backend's shared
#                                memory (mmap, shm) and its own heap both come from it
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
NESTED=${PGSU_NESTED:-none}
case "$NESTED" in none|sublet) ;; *) echo "PGSU_NESTED=$NESTED?" >&2; exit 2 ;; esac
MANAGER=$SCRIPT_DIR/../memory-contexts
MUSL_PORT=$RT/capstone/ports/musl-capstone
MRT=$MUSL_PORT/runtime
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}
ARENA=${PGSU_ARENA_BYTES:-$((64 * 1024 * 1024))}
JOBS=${JOBS:-16}
OPT=${PGSU_OPT_LEVEL:--O1}
case "$OPT" in -O0|-O1|-O2) ;; *) echo "PGSU_OPT_LEVEL=$OPT?" >&2; exit 2;; esac
# PGSU_FROM=runtime|configure|make|link starts at that stage and keeps what the
# earlier ones made: "make" after a patch, "link" after a runtime change (the
# runtime is rebuilt by "runtime", which does not reconfigure).
FROM=${PGSU_FROM:-all}
[[ -f "$TARBALL" ]] || { echo "no tarball $TARBALL" >&2; exit 2; }
[[ $(sha256sum "$TARBALL" | cut -d' ' -f1) == fcb7ab38e23b264d1902cb25e6adafb4525a6ebcbd015434aeef9eda80f528d8 ]] \
  || { echo "not the pinned PostgreSQL 17.5 archive: $TARBALL" >&2; exit 2; }
[[ -f "$MRT/hostcall.c" ]] || { echo "no runtime at $MRT (RUNTIME_REPO=$RT)" >&2; exit 2; }
mkdir -p "$ROOT/runtime" "$ROOT/link" "$ROOT/domain"
log() { echo "[build-domain] $*"; }
# PGSU_ONLY="runtime link" runs just the stages named (a runtime change needs
# no reconfigure: the compiler's objects do not depend on the runtime's).
stage() {
  [[ -z ${PGSU_ONLY:-} ]] || { [[ " $PGSU_ONLY " == *" $1 "* ]]; return; }
  case "$FROM" in all) return 0;; runtime) [[ $1 != musl ]];; configure) [[ $1 == configure || $1 == make || $1 == link ]];; make) [[ $1 == make || $1 == link ]];; link) [[ $1 == link ]];; *) echo "PGSU_FROM=$FROM?" >&2; exit 2;; esac
}

# ---- musl and its archive, private to this build --------------------------
export MUSL_CACHE_ROOT=$ROOT/musl-src
mkdir -p "$MUSL_CACHE_ROOT"
[[ -f "$MUSL_CACHE_ROOT/musl-1.2.5.tar.gz" || ! -f "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" ]] \
  || cp "$CAPSTONE_TMP_ROOT/musl-src/musl-1.2.5.tar.gz" "$MUSL_CACHE_ROOT/"
MUSL=$(bash "$MUSL_PORT/prepare-musl-capstone.sh" | tail -1)
if stage musl; then
  OUT_DIR=$ROOT/musl-build bash "$MUSL_PORT/build-musl-capstone.sh" >/dev/null
fi
ARCHIVE=$ROOT/musl-build/libc-capstone.a
[[ -f "$ARCHIVE" ]] || { echo "no $ARCHIVE" >&2; exit 2; }
log "musl $MUSL, archive $ARCHIVE"

# ---- the runtime: the shared application SDK --------------------------------
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -O1 -w -Wno-int-conversion "${INC[@]}")
RF=("${CF[@]}" -std=c99 -D_XOPEN_SOURCE=700
    -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
if [[ ${CAPSTONE_APPLICATION_PROFILE:-physical} == virtual ]]; then
  CF+=(-mllvm -capstone-gp-free -mllvm -capstone-image-gp)
  RF+=(-mllvm -capstone-gp-free -mllvm -capstone-image-gp)
fi
O=$ROOT/runtime
if stage runtime; then
  EXTRA=()
  [[ $NESTED == sublet ]] && EXTRA=(-DCAPSTONE_APPLICATION_GRANT_BYTES=67108864)
  bash "$RT/capstone/ports/common/application/build-sdk.sh" "$O" "$MUSL" "$ARCHIVE" \
    -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" "${EXTRA[@]}"
fi
export CAPSTONE_SDK=$O
export PATH=$O:$CAPSTONE_LLVM_BIN:$PATH
"$O/capstone-cc" --check-toolchain
{
  printf 'export CAPSTONE_SDK=%q\n' "$O"
  printf 'export PATH=%q:$PATH\n' "$O:$CAPSTONE_LLVM_BIN"
} > "$ROOT/capstone-env.sh"

# ---- the source, patched -------------------------------------------------------
cd "$ROOT/domain"
MODE_FILE=$ROOT/domain/nested.mode
# Over each patch's name and contents, not the paths sha256sum would print:
# the same patch set in a second worktree is the same patch set, and hashing
# the paths made every build root look stale outside the tree that made it.
PATCH_HASH=$({ for p in "$SCRIPT_DIR"/patches/*.patch; do
                 printf '%s ' "$(basename "$p")"; sha256sum < "$p"
               done; } | sha256sum | cut -d' ' -f1)
if [[ -d postgresql-17.5 && $(cat "$ROOT/domain/patchset.sha256" 2>/dev/null) != "$PATCH_HASH" ]]; then
  echo "source patch set changed; use a fresh PG_SU_ROOT" >&2; exit 2
fi
if [[ -f "$MODE_FILE" && $(cat "$MODE_FILE") != "$NESTED" ]]; then
  echo "prepared source is $(cat "$MODE_FILE"), requested $NESTED" >&2; exit 2
fi
if [[ -d postgresql-17.5 && ! -f "$MODE_FILE" && $NESTED != none ]]; then
  echo "existing source has no nested mode identity; use a fresh PG_SU_ROOT" >&2; exit 2
fi
if [[ ! -d postgresql-17.5 ]]; then
  tar xjf "$TARBALL"
  for p in "$SCRIPT_DIR"/patches/*.patch; do
    # This patch targets CheriBSD's external freelist, not the Capstone backend.
    case $p in */0018-*) continue ;; esac
    (cd postgresql-17.5 && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }
    log "applied $(basename "$p")"
  done
  if [[ $NESTED == sublet ]]; then
    for p in "$MANAGER"/patches/postgresql-17.0-000{3,4,5,6,7}-*.patch; do
      (cd postgresql-17.5 && patch --batch --forward --fuzz=0 -p1 -s < "$p") \
        || { echo "nested patch $p did not apply" >&2; exit 2; }
      log "applied $(basename "$p") to PostgreSQL 17.5"
    done
  fi
fi
# The corpus's own module, copied in rather than kept in the tarball: cases 05
# and 06 are reached by a C caller because no SQL statement can reach them, and
# a module is how a C caller gets to run inside the backend. Copied on every
# run so an edit to it rebuilds, and harmless on a reused root.
mkdir -p postgresql-17.5/contrib/pgcorpus_reach
cp "$SCRIPT_DIR"/reach-module/* postgresql-17.5/contrib/pgcorpus_reach/

printf '%s\n' "$NESTED" > "$MODE_FILE"
printf '%s\n' "$PATCH_HASH" > "$ROOT/domain/patchset.sha256"
cd postgresql-17.5

# ---- configure, with the answers this target needs forced -------------------------
if stage configure; then
  [[ -f config.status ]] && make distclean >/dev/null 2>&1 || true
  nested_flags=
  if [[ $NESTED == sublet ]]; then
    nested_flags="-I$MANAGER/src/allocators/sublet -I$RT/capstone/runtime/include"
  elif [[ ${PGSU_GAP_OBSERVER:-0} == 1 ]]; then
    nested_flags="-DPG_SPATIAL_GAP_OBSERVER=1 -I$SCRIPT_DIR/toolchain"
  fi
  CC=capstone-cc CFLAGS="$OPT -DWAIT_USE_POLL $nested_flags" pgac_cv_computed_goto=no \
    ./configure --host=riscv64-unknown-linux-musl --enable-depend \
      --without-readline --without-zlib --without-icu --with-system-tzdata=/usr/share/zoneinfo \
      > "$ROOT/domain-configure.log" 2>&1 \
    || { tail -5 "$ROOT/domain-configure.log" >&2; echo "configure failed" >&2; exit 2; }
  grep -q '^#define MAXIMUM_ALIGNOF 8$' src/include/pg_config.h \
    || { echo "pg_config.h: MAXIMUM_ALIGNOF is not the 8 this forces to 16; read it" >&2; exit 2; }
  sed -i 's/^#define MAXIMUM_ALIGNOF 8$/#define MAXIMUM_ALIGNOF 16 \/* capstone: a capability, not long\/double, sets it *\//' src/include/pg_config.h
  grep -q 'define HAVE_COMPUTED_GOTO' src/include/pg_config.h \
    && { echo "pg_config.h still defines HAVE_COMPUTED_GOTO" >&2; exit 2; }
  grep -E '^#define (MAXIMUM_ALIGNOF|SIZEOF_VOID_P|USE_SYSV_SHARED_MEMORY)' src/include/pg_config.h | sed 's/^/[build-domain] /'
fi

# The modules linked into the image, because a domain cannot dlopen (patch 0015,
# toolchain/static_modules.c): name:directory, the directory relative to the
# source root. The first two are what initdb's setup SQL needs. The contrib
# three carry defects this corpus measures and declare no SHLIB_LINK, so they
# cross-compile with the backend's own settings and need nothing new.
#
# pgcrypto is deliberately not here. Its 17.5 OBJS list openssl.o and
# pgp-mpi-openssl.o unconditionally and its Makefile adds -lcrypto, and no
# OpenSSL is cross-built for capstone64, so the module cannot link into a
# domain image. Its case is measured on the CheriBSD arm only.
MODULES="dict_snowball:src/backend/snowball plpgsql:src/pl/plpgsql/src"
MODULES="$MODULES ltree:contrib/ltree pg_trgm:contrib/pg_trgm"
MODULES="$MODULES fuzzystrmatch:contrib/fuzzystrmatch"
MODULES="$MODULES pgcorpus_reach:contrib/pgcorpus_reach"
module_objs() { make -s -C "$1" -f Makefile -f "$SCRIPT_DIR/toolchain/print-objs.mk" print-objs; }

# ---- make ---------------------------------------------------------------------------
if stage make; then
  # --enable-depend tracks headers from now on; a tree configured without it
  # (before 2026-09-24 20:30) rebuilds nothing for a patched header. PGSU_CLEAN=1
  # starts the objects over once.
  if [[ "${PGSU_CLEAN:-0}" == 1 ]]; then
    # src/include too: its header-stamp would otherwise say the generated
    # headers (errcodes.h, fmgroids.h) are current after their files are gone.
    for d in src/port src/common src/backend src/timezone src/include; do make -C "$d" clean > /dev/null 2>&1 || true; done
    log "objects removed (PGSU_CLEAN=1)"
  fi
  make -C src/backend generated-headers > "$ROOT/domain-genheaders.log" 2>&1
  export CAPSTONE_COMPILE_LOG=$ROOT/domain-objects.tsv
  : > "$CAPSTONE_COMPILE_LOG"
  make -j"$JOBS" -C src/port libpgport_srv.a > "$ROOT/domain-port.log" 2>&1
  make -j"$JOBS" -C src/common libpgcommon_srv.a > "$ROOT/domain-common.log" 2>&1
  make -k -j"$JOBS" -C src/backend > "$ROOT/domain-make.log" 2>&1 || true
  # The two bison parsers crash the compiler at -O2 (C-52's shape); at -O0 they compile.
  for f in parser/gram utils/adt/jsonpath_gram; do
    if [[ ! -f src/backend/$f.o ]]; then
      log "$f.o: retrying at -O0"
      make -C "src/backend/$(dirname "$f")" "$(basename "$f").o" CFLAGS="-O0 -DWAIT_USE_POLL" > /dev/null 2>&1 || true
    fi
  done
  # A directory whose make stopped on an error has no objfiles.txt, and the
  # retried objects alone do not write one; a second pass over the finished
  # objects does (without it, utils/adt's 20 symbols were undefined at the link).
  make -k -j"$JOBS" -C src/backend >> "$ROOT/domain-make.log" 2>&1 || true
  # The upstream top-level make also links postgres before our module table
  # and nested backend are attached, so its status is not a compile gate.
  # All 69 objfiles lists of the pinned 17.5 backend must exist instead.
  listings=$(find src/backend -name objfiles.txt | wc -l)
  [[ $listings -eq 69 ]] \
    || { echo "backend build incomplete: $listings/69 object lists" >&2; exit 2; }
  # The loadable modules (patch 0015), their objects only: a module's own link would
  # be a shared library. Pg_magic_func and _PG_init are renamed per module, so two
  # modules can be linked into one image.
  for md in $MODULES; do
    m=${md%%:*} d=${md#*:}
    # shellcheck disable=SC2046
    # CFLAGS_SL= drops the -fPIC -fvisibility=hidden a contrib MODULE_big adds
    # for a shared library. Hidden symbols are local, and the table below is
    # built from the global ones, so with it left in place a module links into
    # the image with nothing dfmgr can resolve.
    make -k -j"$JOBS" -C "$d" CFLAGS_SL= COPT="-DPg_magic_func=${m}_Pg_magic_func -D_PG_init=${m}__PG_init" \
      $(module_objs "$d") >> "$ROOT/domain-modules.log" 2>&1 || true
  done
  # CREATE EXTENSION reads <name>.control and the version scripts from the share
  # directory and fails there, before it ever asks dfmgr for the library, so the
  # objects alone are not enough: a run whose share directory lacks these files
  # records a silent arm for a statement that never executed. Staged here, and
  # the runner copies this tree into the guest's share directory.
  EXTDIR=$ROOT/share/extension
  rm -rf "$EXTDIR"; mkdir -p "$EXTDIR"
  for md in $MODULES; do
    d=${md#*:}
    case $d in contrib/*) ;; *) continue ;; esac
    m=${md%%:*}
    [[ -f $d/$m.control ]] || { echo "module $m: no $d/$m.control" >&2; exit 2; }
    cp "$d/$m.control" "$EXTDIR/"
    cp "$d"/"$m"--*.sql "$EXTDIR/"
  done
  log "extension files staged in $EXTDIR: $(find "$EXTDIR" -type f | wc -l)"
  # The survey log names an object as make named it, relative to its own directory;
  # what counts is whether the backend's link inputs exist, so look there.
  missing=$(find src/backend src/timezone -name objfiles.txt -exec cat {} + | tr ' ' '\n' | grep -v '^$' | while read -r o; do [[ -f $o ]] || echo "$o"; done)
  [[ -z $missing ]] \
    || { echo "backend build incomplete: missing objects $missing" >&2; exit 2; }
  log "objects compiled: $(find src/backend -name '*.o' | wc -l); link inputs missing: ${missing:-none}"
  if [[ $NESTED == sublet ]]; then
    GAP_FLAGS=()
    if [[ ${PGSU_GAP_OBSERVER:-0} == 1 ]]; then GAP_FLAGS=(-DPG_REUSE_GAP_OBSERVER=1); fi
    "$CAPSTONE_CLANG" "${CF[@]}" -I"$RT/capstone/runtime/include" \
      -I"$MANAGER/src/allocators/sublet" \
      "${GAP_FLAGS[@]}" \
      -c "$MANAGER/src/allocators/sublet/context-pools.c" \
      -o "$ROOT/link/context-pools.o"
    log "Sublet context-pool backend: $ROOT/link/context-pools.o"
  elif [[ ${PGSU_GAP_OBSERVER:-0} == 1 ]]; then
    "$CAPSTONE_CLANG" "${CF[@]}" -I"$SCRIPT_DIR/toolchain" \
      -I"$SCRIPT_DIR/../../../experiments/study" \
      -c "$SCRIPT_DIR/toolchain/spatial-reuse-gap.c" \
      -o "$ROOT/link/spatial-reuse-gap.o"
    log "original-layout context observer: $ROOT/link/spatial-reuse-gap.o"
  fi
fi

# ---- link ---------------------------------------------------------------------------
# What the backend's own link takes: every subdirectory's objfiles.txt (paths from the
# source root), then the two server archives; the port runtime and musl are here.
objs=$(find src/backend src/timezone -name objfiles.txt -exec cat {} + | tr ' ' '\n' | grep -v '^$' | sort -u)
[[ -n "$objs" ]] || { echo "no objfiles.txt under src/backend; make did not get far" >&2; exit 2; }
# The modules, and the table toolchain/static_modules.c resolves them from: every
# global function each one defines, under the name dlsym would be asked for.
NM=$(dirname "$CAPSTONE_CLANG")/llvm-nm
TABLE=$ROOT/link/static_modules_table.h
: > "$TABLE"
for md in $MODULES; do
  m=${md%%:*} d=${md#*:}
  mo=$(module_objs "$d" | tr ' ' '\n' | grep -v '^$' | sed "s|^|$d/|")
  for o in $mo; do [[ -f $o ]] || { echo "module $m: $o was not built (domain-modules.log)" >&2; exit 2; }; done
  echo "PGSU_MODULE($m)" >> "$TABLE"
  # shellcheck disable=SC2086
  "$NM" --defined-only -g $mo | awk '$2 == "T" { print $3 }' | sort -u | while read -r sym; do
    case $sym in ${m}_Pg_magic_func) n=Pg_magic_func ;; ${m}__PG_init) n=_PG_init ;; *) n=$sym ;; esac
    echo "PGSU_SYMBOL($m, \"$n\", $sym)"
  done >> "$TABLE"
  echo "PGSU_MODULE_END" >> "$TABLE"
  grep -q "PGSU_SYMBOL($m, \"Pg_magic_func\"" "$TABLE" || { echo "module $m: no Pg_magic_func" >&2; exit 2; }
  objs="$objs $mo"
done
"$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -I"$ROOT/link" -c "$SCRIPT_DIR/toolchain/static_modules.c" \
  -o "$ROOT/link/static_modules.o"
log "modules linked in: $(grep -c PGSU_MODULE_END "$TABLE"), $(grep -c PGSU_SYMBOL "$TABLE") functions"
EXTRA=()
if [[ $NESTED == sublet ]]; then
  EXTRA+=("$ROOT/link/context-pools.o" -O1 -DEXP_PG_CONTEXT_SUBLET
    -Wl,--wrap=main,--wrap=__capstone_region -I"$RT/capstone/runtime/include"
    "$RT/capstone/ports/common/application/regions.c"
    "$RT/capstone/ports/common/application/initialize.c")
elif [[ ${PGSU_GAP_OBSERVER:-0} == 1 ]]; then
  EXTRA+=("$ROOT/link/spatial-reuse-gap.o")
fi
# shellcheck disable=SC2086
"$O/capstone-cc" -o "$ROOT/link/postgres.dom" "$ROOT/link/static_modules.o" \
  "${EXTRA[@]}" $objs src/port/libpgport_srv.a src/common/libpgcommon_srv.a \
  > "$ROOT/link/link.log" 2>&1 || { tail -20 "$ROOT/link/link.log" >&2; exit 2; }
log "delegated application $ROOT/link/postgres.dom"
