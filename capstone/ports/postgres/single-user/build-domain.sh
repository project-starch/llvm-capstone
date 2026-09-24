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
#   CFLAGS -O2, no -g            Assignment Tracking (C-50) runs only with debug info
#   -DWAIT_USE_POLL              the latch on poll + a self-pipe, not epoll + signalfd
#   pgac_cv_computed_goto=no     a table of &&labels loads untagged (the Lua/CPython trap)
#   MAXIMUM_ALIGNOF 16           configure derives it from long/double, not pointers;
#                                every palloc and tuple would put capabilities on 8-byte slots
#   patches/0001                 the executor's ExprEvalStep guard, for 16-byte pointers
#   patches/0002                 aset's smallest chunk holds a capability (the mmgr port's)
#   patches/0003                 bootstrap and single-user input from a named file
#   level0 arena                 PGSU_ARENA_BYTES, default 64 MiB: the backend's shared
#                                memory (mmap, shm) and its own heap both come from it
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
MUSL_PORT=$RT/capstone/ports/musl-capstone
MRT=$MUSL_PORT/runtime
TARBALL=${1:-$CAPSTONE_TMP_ROOT/pg-mmgr-host/pg.tar.bz2}
ARENA=${PGSU_ARENA_BYTES:-$((64 * 1024 * 1024))}
JOBS=${JOBS:-16}
# PGSU_FROM=runtime|configure|make|link starts at that stage and keeps what the
# earlier ones made: "make" after a patch, "link" after a runtime change (the
# runtime is rebuilt by "runtime", which does not reconfigure).
FROM=${PGSU_FROM:-all}
[[ -f "$TARBALL" ]] || { echo "no tarball $TARBALL" >&2; exit 2; }
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

# ---- the runtime, as musl-capstone/libc-test/build-libc-test.sh builds it ----
INC=(-nostdinc -isystem "$MUSL/arch/capstone64" -isystem "$MUSL/arch/generic"
     -isystem "$MUSL/obj/include" -isystem "$MUSL/include")
CF=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
    -Xclang -target-feature -Xclang +a -ffreestanding -fno-builtin -fno-jump-tables
    -ffunction-sections -fdata-sections -O1 -w -Wno-int-conversion "${INC[@]}")
RF=("${CF[@]}" -std=c99 -D_XOPEN_SOURCE=700
    -I"$MUSL/src/include" -I"$MUSL/src/internal" -I"$MUSL/obj/src/internal")
O=$ROOT/runtime
if stage runtime; then
  rm -f "$O"/*.o
  for s in start-musl set_thread_area setjmp; do
    "$CAPSTONE_CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
      -ffreestanding -O0 -c "$MRT/$s.S" -o "$O/$s.o"
  done
  for f in hostcall tls; do
    "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/$f.c" -o "$O/$f.o"
  done
  "$CAPSTONE_CLANG" "${RF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="($ARENA)" -c "$MRT/level0.c" -o "$O/level0.o"
  source "$MRT/libc_overrides.sh"
  build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
  CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$RT/compiler-rt/lib/builtins
  COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
                -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
  source "$RT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
  # int128 arithmetic (numeric.c's sqrt_var divides an int128) is a builtin
  # call on this target; these are the softfloat set's integer counterparts.
  for b in divti3 udivti3 modti3 umodti3 udivmodti4 multi3 ashlti3 lshrti3; do
    "$CAPSTONE_CLANG" "${COMMON_FLAGS[@]}" -c "$RT/compiler-rt/lib/builtins/$b.c" -o "$O/int128-$b.o"
  done
  "$CAPSTONE_CLANG" "${CF[@]}" -std=c11 -O1 -c "$SCRIPT_DIR/toolchain/domain_entry.c" -o "$O/domain_entry.o"
  log "runtime: $(ls "$O"/*.o | wc -l) objects from $MRT, level0 arena $ARENA bytes"
fi

# ---- the compiler configure and make are given ------------------------------
export PGSU_MUSL=$MUSL PGSU_RUNTIME_DIR=$O PGSU_LIBC_ARCHIVE=$ARCHIVE
export PGSU_LINKER_SCRIPT=$RT/capstone/my_first_domain/link.ld
export PATH=$SCRIPT_DIR/toolchain:$PATH
{
  echo "export CAPSTONE_CLANG='$CAPSTONE_CLANG' CAPSTONE_LD_LLD='$CAPSTONE_LD_LLD'"
  echo "export PGSU_MUSL='$PGSU_MUSL' PGSU_RUNTIME_DIR='$PGSU_RUNTIME_DIR' PGSU_LIBC_ARCHIVE='$PGSU_LIBC_ARCHIVE'"
  echo "export PGSU_LINKER_SCRIPT='$PGSU_LINKER_SCRIPT'"
  echo "export PATH='$SCRIPT_DIR/toolchain':\$PATH"
} > "$ROOT/capstone-env.sh"

# ---- the source, patched -------------------------------------------------------
cd "$ROOT/domain"
if [[ ! -d postgresql-17.5 ]]; then
  tar xjf "$TARBALL"
  for p in "$SCRIPT_DIR"/patches/*.patch; do
    (cd postgresql-17.5 && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }
    log "applied $(basename "$p")"
  done
fi
cd postgresql-17.5

# ---- configure, with the answers this target needs forced -------------------------
if stage configure; then
  [[ -f config.status ]] && make distclean >/dev/null 2>&1 || true
  CC=capstone-cc CFLAGS="-O2 -DWAIT_USE_POLL" pgac_cv_computed_goto=no \
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
  export PGSU_SURVEY_LOG=$ROOT/domain-objects.tsv
  : > "$PGSU_SURVEY_LOG"
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
  # The survey log names an object as make named it, relative to its own directory;
  # what counts is whether the backend's link inputs exist, so look there.
  missing=$(find src/backend src/timezone -name objfiles.txt -exec cat {} + | tr ' ' '\n' | grep -v '^$' | while read -r o; do [[ -f $o ]] || echo "$o"; done)
  log "objects compiled: $(find src/backend -name '*.o' | wc -l); link inputs missing: ${missing:-none}"
fi

# ---- link ---------------------------------------------------------------------------
# What the backend's own link takes: every subdirectory's objfiles.txt (paths from the
# source root), then the two server archives; the port runtime and musl are here.
objs=$(find src/backend src/timezone -name objfiles.txt -exec cat {} + | tr ' ' '\n' | grep -v '^$' | sort -u)
[[ -n "$objs" ]] || { echo "no objfiles.txt under src/backend; make did not get far" >&2; exit 2; }
# shellcheck disable=SC2086
"$CAPSTONE_LD_LLD" --gc-sections -T "$PGSU_LINKER_SCRIPT" -o "$ROOT/link/postgres.dom" \
  "$O"/*.o $objs src/port/libpgport_srv.a src/common/libpgcommon_srv.a "$ARCHIVE" \
  > "$ROOT/link/link.log" 2>&1 && rc=0 || rc=$?
{ grep -oE 'undefined symbol: [^ ]+' "$ROOT/link/link.log" || true; } | sed 's/undefined symbol: //' | sort -u > "$ROOT/link/undefined.txt"
{ grep -E 'error:' "$ROOT/link/link.log" | grep -v 'undefined symbol' || true; } | head -5 | sed 's/^/[build-domain] /'
log "link rc=$rc, $(wc -l < "$ROOT/link/undefined.txt") undefined symbols (link/undefined.txt)"
if [[ $rc -ne 0 ]]; then
  # A lower-bound image with the unresolved symbols ignored, to size it and to try.
  "$CAPSTONE_LD_LLD" --gc-sections --unresolved-symbols=ignore-all -T "$PGSU_LINKER_SCRIPT" \
    -o "$ROOT/link/postgres-lowerbound.dom" "$O"/*.o $objs src/port/libpgport_srv.a \
    src/common/libpgcommon_srv.a "$ARCHIVE" > "$ROOT/link/link-lowerbound.log" 2>&1 || true
fi
ls -la "$ROOT/link"/*.dom 2>/dev/null | awk '{print "[build-domain] " $5 " " $9}'
