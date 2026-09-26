#!/usr/bin/env bash
# perl as a Capstone musl domain: musl and the port runtime from a given
# llvm-capstone tree, perl at a pinned release cross-built with perl-cross, with
# this port's patches, and linked as a domain. The same release is built natively
# as the reference its output is compared with.
#
#   CAPSTONE_LLVM_BUILD_DIR=<llvm build> [RUNTIME_REPO=<llvm-capstone tree>] \
#     bash build-perl-domain.sh
#
# Outputs, under $PERLD_ROOT (default $CAPSTONE_TMP_ROOT/perl-domain):
#   src/perl-<version>/perl          the interpreter, a domain image
#   native/bin/perl                  the same release built natively, the reference
#
# WHY perl-cross AND NOT perl's OWN Configure: Configure runs target programs to
# answer its questions, which a cross build cannot do. perl-cross answers them by
# compiling only -- sizes come from the ELF symbol table via readelf
# (cnf/configure_type.sh checksize) -- and builds miniperl with the HOST compiler
# (its Makefile's miniperl rule), so no target binary is ever executed. That is the
# same host/target split the PostgreSQL port uses for its build tools.
#
# Knobs: PERLD_FROM=musl|runtime|perl starts at that stage. PERLD_HEAP=level0
# (default) or sublet picks the domain's malloc, as the mruby port's does.
# PERL_MIRROR=<dir with the tarballs> and PERL_CROSS_MIRROR=<a perl-cross clone>
# avoid the network.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null
ROOT=${PERLD_ROOT:-$CAPSTONE_TMP_ROOT/perl-domain}
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
MUSL_PORT=$RT/capstone/ports/musl-capstone
MRT=$MUSL_PORT/runtime
ARENA=${PERLD_ARENA_BYTES:-$((64 * 1024 * 1024))}
JOBS=${JOBS:-16}
FROM=${PERLD_FROM:-all}
HEAP=${PERLD_HEAP:-level0}
HEAP_LOG=${PERLD_HEAP_LOG:-26}
case "$HEAP" in level0|sublet) ;; *) echo "PERLD_HEAP=$HEAP? (level0, sublet)" >&2; exit 2 ;; esac

# The pin. 5.36.3 is the evaluation's release (docs/design/perl-sublet-port-evaluation.md):
# the SV arena mechanism is the same in every release perl-cross supports, and 5.36
# is the oldest that has the current file layout, so one patch serves 5.36 to 5.44.
PERL_VERSION=${PERL_VERSION:-5.36.3}
PERL_URL=https://www.cpan.org/src/5.0/perl-$PERL_VERSION.tar.gz
PERL_CROSS_URL=https://github.com/arsv/perl-cross.git
PERL_CROSS_COMMIT=c2d8f8b7027ed20cd982c9f2c091463510b89f33
PATCHES=$SCRIPT_DIR/patches/$PERL_VERSION
[[ -d "$PATCHES" ]] || { echo "no patches/$PERL_VERSION" >&2; exit 2; }

[[ -f "$MRT/hostcall.c" ]] || { echo "no runtime at $MRT (RUNTIME_REPO=$RT)" >&2; exit 2; }
mkdir -p "$ROOT/runtime" "$ROOT/src" "$ROOT/dl"
log() { echo "[build-perl] $*"; }
stage() {
  case "$FROM" in all) return 0;; runtime) [[ $1 != musl ]];; perl) [[ $1 == perl ]];;
    *) echo "PERLD_FROM=$FROM?" >&2; exit 2;; esac
}

# ---- musl and its archive, private to this build ----------------------------
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

# ---- the runtime, as the mruby and PostgreSQL ports build it ----------------
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
  HCF=(); EF=()
  [[ $HEAP == sublet ]] && { HCF=(-DCAPSTONE_PROGRAM_REGIONS=1); EF=(-DPERLD_SUBLET_HEAP=1); }
  "$CAPSTONE_CLANG" "${RF[@]}" "${HCF[@]}" -c "$MRT/hostcall.c" -o "$O/hostcall.o"
  "$CAPSTONE_CLANG" "${RF[@]}" -c "$MRT/tls.c" -o "$O/tls.o"
  if [[ $HEAP == sublet ]]; then
    "$CAPSTONE_CLANG" "${RF[@]}" -I"$RT/capstone/sublet" -DCAPSTONE_SUBLET_HEAP_LOG="$HEAP_LOG" \
      -c "$MRT/sublet_heap.c" -o "$O/heap.o"
  else
    "$CAPSTONE_CLANG" "${RF[@]}" -DCAPSTONE_LEVEL0_ARENA_BYTES="($ARENA)" -c "$MRT/level0.c" -o "$O/level0.o"
  fi
  source "$MRT/libc_overrides.sh"
  build_musl_overrides "$CAPSTONE_CLANG" "$O" "$MUSL" "${RF[@]}"
  CLANG=$CAPSTONE_CLANG OBJ_DIR=$O COMPILER_RT=$RT/compiler-rt/lib/builtins
  COMMON_FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
                -ffreestanding -fno-builtin -ffunction-sections -fdata-sections -O1 -w)
  source "$RT/capstone/benchmarks/beebs/build-beebs-softfloat-common.sh"
  "$CAPSTONE_CLANG" "${CF[@]}" "${EF[@]}" -std=c11 -O1 -c "$SCRIPT_DIR/toolchain/domain_entry.c" -o "$O/domain_entry.o"
  echo "$HEAP" > "$O/.heap"
  log "runtime: $(ls "$O"/*.o | wc -l) objects from $MRT, heap $HEAP"
fi
[[ $(cat "$O/.heap" 2>/dev/null) == "$HEAP" ]] \
  || { echo "the runtime in $O was built for PERLD_HEAP=$(cat "$O/.heap" 2>/dev/null); rebuild it (PERLD_FROM=runtime)" >&2; exit 2; }

# ---- the compiler and linker perl-cross is given ---------------------------
export PERLD_MUSL=$MUSL PERLD_RUNTIME_DIR=$O PERLD_LIBC_ARCHIVE=$ARCHIVE
export PERLD_LINKER_SCRIPT=$RT/capstone/my_first_domain/link.ld
export PATH=$SCRIPT_DIR/toolchain:$CAPSTONE_LLVM_BIN:$PATH

# ---- perl, pinned, patched, cross-built ------------------------------------
S=$ROOT/src/perl-$PERL_VERSION
if [[ ! -f "$S/.perld-ready" ]]; then
  rm -rf "$S"
  TAR=${PERL_MIRROR:-$ROOT/dl}/perl-$PERL_VERSION.tar.gz
  [[ -f "$TAR" ]] || curl -sSfL "$PERL_URL" -o "$ROOT/dl/perl-$PERL_VERSION.tar.gz"
  [[ -f "$TAR" ]] || TAR=$ROOT/dl/perl-$PERL_VERSION.tar.gz
  tar xzf "$TAR" -C "$ROOT/src"
  chmod -R u+w "$S"
  # perl-cross is unpacked OVER the perl tree; its configure lives there.
  PC=$ROOT/src/perl-cross
  if [[ ! -d "$PC/.git" ]]; then
    rm -rf "$PC"
    git clone -q "${PERL_CROSS_MIRROR:-$PERL_CROSS_URL}" "$PC"
  fi
  git -C "$PC" checkout -q "$PERL_CROSS_COMMIT"
  cp -a "$PC"/. "$S/"
  touch "$S/.perld-ready"
fi
# OUR patches go on AFTER perl-cross's own, which its Makefile applies at build
# time (its `crosspatch` target, Makefile:61-70) and which touch perl.h as ours
# does: applied first, ours left perl-cross's hunk looking already-applied and the
# build stopped in the patch step.
apply_our_patches() {
  [[ -f "$S/.perld-patched" ]] && return 0
  for p in "$PATCHES"/*.patch; do
    (cd "$S" && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }
    log "applied $(basename "$p")"
  done
  touch "$S/.perld-patched"
}

if stage perl; then
  # The NATIVE reference, the same release with its own Configure, out of tree.
  if [[ ! -x "$ROOT/native/bin/perl" ]]; then
    N=$ROOT/src/native-perl-$PERL_VERSION
    rm -rf "$N"; mkdir -p "$N"
    tar xzf "${PERL_MIRROR:-$ROOT/dl}/perl-$PERL_VERSION.tar.gz" -C "$N" --strip-components=1
    chmod -R u+w "$N"
    (cd "$N" && ./Configure -des -Dprefix="$ROOT/native" -Uusedl -Uusethreads -Uusemymalloc \
        -Dusenm=false -Dinc_version_list=none >/dev/null \
      && make -j"$JOBS" >/dev/null && make install >/dev/null) \
      || { echo "the native reference build failed (see $N)" >&2; exit 2; }
  fi
  log "native reference $("$ROOT/native/bin/perl" -e 'print $]')"

  # The CROSS build. Every value configure cannot probe without running target
  # code, or that the linux hints guess wrongly for a domain, is given here:
  #   alignbytes  16, the capability's alignment; the hints derive 8 from long
  #   d_nanosleep this release's configure leaves the variable unset, and the
  #               config.h template then emits "# HAS_NANOSLEEP", which is not a
  #               directive and stops every compilation
  #   _GNU_SOURCE musl declares memrchr, setresuid, setresgid and eaccess only
  #               under it, while the symbols are in libc either way -- so
  #               configure's link tests find them and the compile would not
  # Two extensions are dropped:
  #   PerlIO/mmap needs PL_mmap_page_size, which -Ud_mmap removes;
  #   Time-HiRes  its Makefile.PL COMPILES AND RUNS a probe (dist/Time-HiRes/
  #               Makefile.PL, "tmp$$"), so a cross build would execute a domain
  #               image on the host. It is the one build step in perl that does.
  #
  #   d_mmap      a domain has none: musl has the symbol, so configure's link test
  #               says yes, and perl then reads the page size at startup for its
  #               mmap PerlIO layer. A domain has no auxv either, so
  #               sysconf(_SC_PAGESIZE) is 0 and perl dies with
  #               "panic: bad pagesize 0" before running any program. Undefined,
  #               perl skips the layer, which nothing here asks for.
  (cd "$S" && ./configure \
      --target=capstone64-unknown-elf --targetarch=capstone64-unknown-elf \
      --with-cc=capstone-cc --with-ranlib=true --with-ar=llvm-ar --with-nm=llvm-nm \
      --with-objdump=llvm-objdump --with-readelf=llvm-readelf --hints=linux \
      -Uusedl -Uusethreads -Uusemymalloc -Ud_nanosleep -Ud_mmap \
      --disable-mod=PerlIO/mmap,Time-HiRes \
      -Dalignbytes=16 -Doptimize="${PERLD_OPT:--O2}" -Accflags=-D_GNU_SOURCE \
      > "$ROOT/configure.log" 2>&1) \
    || { tail -5 "$ROOT/configure.log" >&2; echo "configure failed (see $ROOT/configure.log)" >&2; exit 2; }
  # A malformed config.h line reads as a missing configure answer, and every
  # compilation then fails on it; catch it here rather than 300 errors later.
  python3 - "$S/config.h" <<'PY' || exit 2
import re, sys
ok = ('define','undef','ifdef','ifndef','if','else','elif','endif','include','error','pragma','warning')
bad = [(i, l.rstrip()) for i, l in enumerate(open(sys.argv[1]), 1)
       if (m := re.match(r'^#\s+([A-Za-z_]\w*)', l)) and m.group(1) not in ok]
for i, l in bad:
    print(f"config.h:{i}: not a directive -- a configure answer is missing: {l}")
sys.exit(1 if bad else 0)
PY
  # perl-cross's own patches first, then ours; both are idempotent by stamp file.
  (cd "$S" && make crosspatch > "$ROOT/crosspatch.log" 2>&1) \
    || { tail -5 "$ROOT/crosspatch.log" >&2; echo "make crosspatch failed (see $ROOT/crosspatch.log)" >&2; exit 2; }
  apply_our_patches
  (cd "$S" && make -j"$JOBS" perl > "$ROOT/make.log" 2>&1) \
    || { grep -m5 "error:" "$ROOT/make.log" >&2; echo "make failed (see $ROOT/make.log)" >&2; exit 2; }
  # A size that does not fit its field is silent at run time and fatal: the arena
  # would hand out a slot smaller than the body it is for (patches 0001).
  if grep -q "changes value from" "$ROOT/make.log"; then
    grep -m3 -B2 "changes value from" "$ROOT/make.log" >&2
    echo "a constant was truncated; see the lines above" >&2; exit 2
  fi
  log "domain image $S/perl"
fi
ls -la "$S/perl" "$ROOT/native/bin/perl" 2>/dev/null | awk '{print "[build-perl] " $NF, $5" bytes"}'
