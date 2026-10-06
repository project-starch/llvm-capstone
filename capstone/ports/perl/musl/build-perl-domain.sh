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
# avoid the network. PERLD_SV_HEADS=1 builds the study variant: SV heads come
# from the lifetime adapter in ../sv-heads (its patch, -DPERL_SV_HEAD_ADAPTER,
# and link/perl-sv-heads.o for ../../common/application/build.py --nested perl,
# which is what grants the adapter its region; the image this script leaves behind
# only proves the link resolves). PERLD_SDK_CFLAGS adds C flags to the SDK's own
# -O1 -- the corpus's unprotected arm is PERLD_SDK_CFLAGS=-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0.
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
case "$HEAP" in level0|sublet|sublet-svheads) ;;
  *) echo "PERLD_HEAP=$HEAP? (level0, sublet, sublet-svheads)" >&2; exit 2 ;; esac
# sublet-svheads is CUMULATIVE and is this port's analogue of the mruby port's
# sublet-gc: the Sublet heap for the system allocator AND the SV head arena through
# the lifetime adapter, in one image. It implies PERLD_SV_HEADS=1.
[[ $HEAP == sublet-svheads ]] && SV_HEADS_DEFAULT=1 || SV_HEADS_DEFAULT=0
SV_HEADS=${PERLD_SV_HEADS:-$SV_HEADS_DEFAULT}
case "$SV_HEADS" in 0|1) ;; *) echo "PERLD_SV_HEADS=$SV_HEADS? (0, 1)" >&2; exit 2 ;; esac
[[ $HEAP == sublet-svheads && $SV_HEADS == 0 ]] \
  && { echo "PERLD_HEAP=sublet-svheads is the adapter arm; it cannot take PERLD_SV_HEADS=0" >&2; exit 2; }

# The pin. 5.36.3 is the evaluation's release (docs/design/perl-sublet-port-evaluation.md):
# the SV arena mechanism is the same in every release perl-cross supports, and 5.36
# is the oldest that has the current file layout, so one patch serves 5.36 to 5.44.
PERL_VERSION=${PERL_VERSION:-5.36.3}
PERL_URL=https://www.cpan.org/src/5.0/perl-$PERL_VERSION.tar.gz
PERL_CROSS_URL=https://github.com/arsv/perl-cross.git
PERL_CROSS_COMMIT=c2d8f8b7027ed20cd982c9f2c091463510b89f33
PATCHES=$SCRIPT_DIR/patches/$PERL_VERSION
[[ -d "$PATCHES" ]] || { echo "no patches/$PERL_VERSION" >&2; exit 2; }
PATCH_FILES=("$PATCHES"/*.patch)
SV_HEAD_FLAGS=
if [[ $SV_HEADS == 1 ]]; then
  PATCH_FILES+=("$SCRIPT_DIR"/../sv-heads/patches/$PERL_VERSION/*.patch)
  SV_HEAD_FLAGS=" -DPERL_SV_HEAD_ADAPTER"
fi

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

# ---- shared application SDK (also usable by other upstream build systems) ----
O=$ROOT/runtime
SDK_HEAP=$HEAP
[[ $HEAP == sublet-svheads ]] && SDK_HEAP=sublet
if stage runtime; then
  EXTRA=()
  if [[ $HEAP == sublet-svheads ]]; then
    # One grant, split by regions.c below: the Sublet heap takes region 0 and the
    # SV head adapter region 1. Without the extra 32 MiB the split aborts.
    EXTRA+=(-DCAPSTONE_APPLICATION_GRANT_BYTES="$(( (2 << HEAP_LOG) + (32 << 20) ))")
  fi
  # PERLD_SDK_CFLAGS adds C flags to the SDK's own -O1, for an arm that differs from the
  # default only in how the system allocator bounds its objects. It must arrive as ONE cmake
  # argument: a flag list that is word-split becomes separate -D arguments, and cmake takes
  # an unknown -D as a cache variable and silently compiles without it, so the arm builds
  # and comes out byte-identical to the default one (seen 2026-10-05 on the mruby port's
  # unprotected control, caught only by comparing image hashes). CMAKE_C_FLAGS itself belongs
  # to the domain toolchain (-nostdinc, -isystem ...); overriding that drops the sysroot.
  if [[ -n ${PERLD_SDK_CFLAGS:-} ]]; then
    EXTRA+=(-DCMAKE_C_FLAGS_RELEASE="-O1 ${PERLD_SDK_CFLAGS}")
  fi
  bash "$RT/capstone/ports/common/application/build-sdk.sh" "$O" "$MUSL" "$ARCHIVE" \
    -DCAPSTONE_APPLICATION_HEAP="$SDK_HEAP" -DCAPSTONE_APPLICATION_HEAP_LOG="$HEAP_LOG" \
    -DCAPSTONE_APPLICATION_DATA_BYTES="${PERLD_DATA_BYTES:-33554432}" \
    -DCAPSTONE_APPLICATION_ARENA_BYTES="$ARENA" "${EXTRA[@]}"
  printf '%s\n' "$HEAP ${PERLD_SDK_CFLAGS:-}" > "$O/.heap"
fi
[[ $(cat "$O/.heap" 2>/dev/null) == "$HEAP ${PERLD_SDK_CFLAGS:-}" && -x "$O/capstone-cc" ]] \
  || { echo "rebuild the application SDK (PERLD_FROM=runtime)" >&2; exit 2; }
export CAPSTONE_SDK=$O
export PATH=$O:$CAPSTONE_LLVM_BIN:$PATH
"$O/capstone-cc" --check-toolchain
CC=$(python3 - "$O/sdk.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["cc"])
PY
)
CC_HASH=$(sha256sum "$CC" | cut -d' ' -f1)

# ---- perl, pinned, patched, cross-built ------------------------------------
S=$ROOT/src/perl-$PERL_VERSION
PATCH_HASH=$(sha256sum "${PATCH_FILES[@]}" | sha256sum | cut -d' ' -f1)
if [[ -f "$S/.perld-patched" && $(cat "$S/.perld-patchset" 2>/dev/null) != "$PATCH_HASH" ]]; then
  log "source patch set changed; unpacking Perl again"
  rm -rf "$S"
fi
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
  for p in "${PATCH_FILES[@]}"; do
    (cd "$S" && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }
    log "applied $(basename "$p")"
  done
  touch "$S/.perld-patched"
  printf '%s\n' "$PATCH_HASH" > "$S/.perld-patchset"
}

if stage perl; then
  if [[ -f "$S/Makefile.config" && $(cat "$ROOT/.compiler-hash" 2>/dev/null) != "$CC_HASH" ]]; then
    log "compiler binary changed; cleaning upstream objects"
    (cd "$S" && make clean > "$ROOT/clean.log" 2>&1) \
      || { tail -5 "$ROOT/clean.log" >&2; echo "make clean failed" >&2; exit 2; }
  fi
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
      -Dalignbytes=16 -Doptimize="${PERLD_OPT:--O2}" -Accflags="-D_GNU_SOURCE$SV_HEAD_FLAGS" \
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
  # The study variant's adapter. This link only proves the image resolves; the
  # measured image is relinked by experiments/applications/build.py --nested
  # perl, which also grants the adapter its region.
  MAKE_VARS=()
  if [[ $SV_HEADS == 1 ]]; then
    mkdir -p "$ROOT/link"
    "$O/capstone-cc" -O1 -I"$RT/capstone/runtime/include" \
      -c "$SCRIPT_DIR/../sv-heads/capstone.c" -o "$ROOT/link/perl-sv-heads.o"
    LINK_OBJS=("$ROOT/link/perl-sv-heads.o")
    if [[ $HEAP == sublet-svheads ]]; then
      # The cumulative arm has TWO consumers of the grant, so the single pool is
      # split: PORT_HEAP_REGION_BYTES to the Sublet heap as region 0, the rest to
      # the adapter as region 1. The default wrapper in regions.c returns 0 for
      # index 0, which a Sublet outer heap cannot survive -- the same split the
      # mruby port uses for sublet-gc.
      "$O/capstone-cc" -O1 -I"$RT/capstone/runtime/include" -DEXP_HEAP_AND_POOL \
        -DPORT_HEAP_REGION_BYTES="$((2 << HEAP_LOG))UL" -DPORT_INNER_REGION_BYTES=33554432UL \
        -c "$RT/capstone/ports/common/application/regions.c" -o "$ROOT/link/regions.o"
      LINK_OBJS+=("$ROOT/link/regions.o" "-Wl,--wrap=__capstone_region")
    fi
    MAKE_VARS=("LIBS=${LINK_OBJS[*]}")
  fi
  # The upstream Makefile cannot see the external SDK archive dependencies.
  # Relink on each requested Perl build; compiled upstream objects remain reusable.
  rm -f "$S/perl"
  (cd "$S" && make -j"$JOBS" perl "${MAKE_VARS[@]}" > "$ROOT/make.log" 2>&1) \
    || { grep -m5 "error:" "$ROOT/make.log" >&2; echo "make failed (see $ROOT/make.log)" >&2; exit 2; }
  # A size that does not fit its field is silent at run time and fatal: the arena
  # would hand out a slot smaller than the body it is for (patches 0001).
  if grep -q "changes value from" "$ROOT/make.log"; then
    grep -m3 -B2 "changes value from" "$ROOT/make.log" >&2
    echo "a constant was truncated; see the lines above" >&2; exit 2
  fi
  log "domain image $S/perl"
  printf '%s\n' "$CC_HASH" > "$ROOT/.compiler-hash"
fi

# ---- the runnable library ----
# perl-cross populates the TARGET lib/ with almost nothing: 639 of the pure-Perl
# modules the same release installs natively are absent, XSLoader.pm and
# DynaLoader.pm among them. That matters even though every extension here is
# linked statically (-Uusedl), because perl still reaches an XS layer through its
# .pm: PerlIO_find_layer("scalar") does `require PerlIO::scalar`, that does
# `XSLoader::load`, and with no XSLoader.pm the require fails SILENTLY --
# `open $fh, '>', \$str` then succeeds with the generic `perlio` layer pushed
# instead of `scalar`, writes are discarded, and the backing scalar stays undef.
# Measured 2026-10-06: the domain reported `LAYERS perlio` where native reports
# `LAYERS scalar`, which cost a corpus case its verdict (bug-corpora/perl/
# release-differential, 10_254b30e378) and would silently degrade any workload
# needing a module. The native reference is the same release built by this script,
# so its lib/ is the right source; the cross tree's own copy of a file always wins,
# which keeps the target's Config.pm.
if [[ -d $ROOT/native/lib/$PERL_VERSION && -d $S/lib ]]; then
  # TWO source roots, and the second is easy to miss: perl installs the .pm stub of
  # an XS module under lib/<ver>/<archname>/, not at the root, so a walk of the
  # version directory alone copies DynaLoader.pm and PerlIO/scalar.pm to
  # lib/<archname>/... where @INC never looks -- the fill then reports hundreds of
  # files and still leaves the two that matter missing. Flattening them is correct
  # here: their content is architecture-independent (byte-identical to the host
  # build's own copies), and the XS they front is already linked into the image.
  # Ask the native perl for its archname rather than guessing the directory: a
  # name test picks up Net/ as well, because Net::Config exists, and Net/*.pm
  # would then be flattened into the library root under the wrong names.
  ARCH=$("$ROOT/native/bin/perl" -MConfig -e 'print $Config{archname}')
  [[ -n $ARCH ]] || { echo "cannot read the native reference's archname" >&2; exit 2; }
  SRCS=("$ROOT/native/lib/$PERL_VERSION")
  [[ -d $ROOT/native/lib/$PERL_VERSION/$ARCH ]] && SRCS+=("$ROOT/native/lib/$PERL_VERSION/$ARCH")
  filled=0
  for src in "${SRCS[@]}"; do
    while IFS= read -r rel; do
      if [[ ! -e $S/lib/$rel ]]; then
        mkdir -p "$S/lib/$(dirname "$rel")"
        cp "$src/$rel" "$S/lib/$rel"
        filled=$((filled + 1))
      fi
    done < <(cd "$src" && find . \( -name '*.pm' -o -name '*.pl' \) \
               -not -path './unicore/*' -not -path "./$ARCH/*" -printf '%P\n')
  done
  log "library: filled $filled module(s) the cross build did not install"
  for must in XSLoader.pm DynaLoader.pm PerlIO/scalar.pm; do
    [[ -e $S/lib/$must ]] \
      || { echo "library staging missed $must; an XS layer cannot be registered without it" >&2; exit 2; }
  done
  # A positive control on the result, because the failure mode is silence: the
  # layer the in-memory filehandle pushes must be `scalar`, not `perlio`.
  "$ROOT/native/bin/perl" -e 'open my $fh, ">", \my $s or die; print $fh "x";
    die "native reference cannot do in-memory files" if !defined $s' \
    || { echo "the native reference itself cannot open an in-memory file" >&2; exit 2; }
fi
ls -la "$S/perl" "$ROOT/native/bin/perl" 2>/dev/null | awk '{print "[build-perl] " $NF, $5" bytes"}'
