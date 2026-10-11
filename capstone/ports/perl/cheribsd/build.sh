#!/usr/bin/env bash
# Complete Perl 5.36.3, cross-built for CheriBSD purecap. PERL_CHERI_SV_HEADS=1
# builds the study variant: SV heads from the PoisonCap lifetime adapter in
# ../sv-heads, and the common CheriBSD phase observer.
#
# PERL_CHERI_STATIC=1 links the interpreter statically. A dynamically linked
# purecap perl is refused by the loader on the stock CheriBSD image with
# "Traditional TLS not supported", before main -- measured 2026-10-06 on both an
# earlier dynamic build and the recipe build, with and without a libc preload.
# The mruby port links static for the same reason (../mruby/cheribsd/build.sh).
# The dynamic image only ever ran on the PoisonCap platform, whose own sshd dies
# on a poison exception part way through a file copy, so it cannot carry a corpus.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
: "${CHERI_SDK:?set CHERI_SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT}"
ROOT=${PERL_CHERI_ROOT:-$CAPSTONE_TMP_ROOT/perl-cheribsd}
SV_HEADS=${PERL_CHERI_SV_HEADS:-0}
[[ $SV_HEADS == 0 || $SV_HEADS == 1 ]] || { echo "PERL_CHERI_SV_HEADS must be 0 or 1" >&2; exit 2; }
PATCH_FILES=("$HERE"/../musl/patches/5.36.3/*.patch)
CCFLAGS=
if [[ $SV_HEADS == 1 ]]; then
  PATCH_FILES+=("$HERE"/../sv-heads/patches/5.36.3/*.patch)
  CCFLAGS=-DPERL_SV_HEAD_ADAPTER
fi
ARCHIVE=${PERL_ARCHIVE:-$CAPSTONE_TMP_ROOT/perl-src/perl-5.36.3.tar.gz}
CROSS_REV=c2d8f8b7027ed20cd982c9f2c091463510b89f33
ARCHIVE_SHA=f2a1ad88116391a176262dd42dfc52ef22afb40f4c0e9810f15d561e6f1c726a
[[ ! -e $ROOT ]] || { echo "build root already exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/src" "$(dirname "$ARCHIVE")"
[[ -f $ARCHIVE ]] || curl -fL https://www.cpan.org/src/5.0/perl-5.36.3.tar.gz -o "$ARCHIVE"
[[ $(sha256sum "$ARCHIVE" | cut -d' ' -f1) == "$ARCHIVE_SHA" ]] \
  || { echo 'Perl archive SHA-256 mismatch' >&2; exit 2; }
CROSS=${PERL_CROSS_MIRROR:-$ROOT/perl-cross}
if [[ ! -d $CROSS/.git ]]; then
  git clone -q https://github.com/arsv/perl-cross.git "$CROSS"
fi
S=$ROOT/src/perl-5.36.3
tar -xzf "$ARCHIVE" -C "$ROOT/src"
chmod -R u+w "$S"
git -C "$CROSS" archive "$CROSS_REV" | tar -xf - -C "$S"
# Configure accepts a compiler path. Keep the target arguments in this wrapper.
python3 - "$ROOT/cc" "$CHERI_SDK" "$CHERI_SYSROOT" <<'PY'
from pathlib import Path
import shlex, sys
path, sdk, sysroot = map(Path, sys.argv[1:])
argv = [str(sdk / 'bin/clang'), '--target=riscv64-unknown-freebsd13',
        '--sysroot=' + str(sysroot), '-march=rv64imafdcxcheri', '-mabi=l64pc128d',
        '-mno-relax', '-fuse-ld=lld', '-B' + str(sdk / 'bin')]
path.write_text('#!/bin/sh\nexec ' + shlex.join(argv) + ' "$@"\n')
path.chmod(0o755)
PY
(
  cd "$S"
  ./configure --target=riscv64-unknown-freebsd --targetarch=riscv64-freebsd-purecap \
    --with-cc="$ROOT/cc" --with-ar="$CHERI_SDK/bin/llvm-ar" \
    --with-nm="$CHERI_SDK/bin/llvm-nm" --with-ranlib="$CHERI_SDK/bin/llvm-ranlib" \
    --with-objdump="$CHERI_SDK/bin/llvm-objdump" --with-readelf="$CHERI_SDK/bin/llvm-readelf" \
    -Uusedl -Uusethreads -Uusemymalloc --disable-mod=Time-HiRes \
    -Dosname=freebsd -Darchname=riscv64-freebsd-purecap \
    -Dd_nanosleep=define -Dalignbytes=16 -Doptimize=-O1 ${CCFLAGS:+-Accflags=$CCFLAGS} \
    ${PERL_CHERI_STATIC:+-Aldflags=-static} \
    > "$ROOT/configure.log" 2>&1
  make crosspatch > "$ROOT/crosspatch.log" 2>&1
  for patch_file in "${PATCH_FILES[@]}"; do
    patch --batch --fuzz=0 -p1 -s < "$patch_file"
  done
  MAKE_VARS=()
  if [[ $SV_HEADS == 1 ]]; then
    # The adapter and the phase observer join the final link only; the host
    # miniperl never sees the define.
    COMMON=(--target=riscv64-unknown-freebsd13 --sysroot="$CHERI_SYSROOT"
            -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -O1)
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" -I"$HERE/../../common/include" \
      -c "$HERE/../sv-heads/cheribsd.c" -o "$ROOT/perl-sv-heads.o"
    "$CHERI_SDK/bin/clang" "${COMMON[@]}" \
      -c "$HERE/../../../experiments/applications/cheribsd-memory.c" \
      -o "$ROOT/cheribsd-memory.o"
    MAKE_VARS=("LIBS=$ROOT/perl-sv-heads.o $ROOT/cheribsd-memory.o -Wl,--wrap=main,--wrap=write")
  fi
  make -j"${JOBS:-10}" perl "${MAKE_VARS[@]}" > "$ROOT/make.log" 2>&1 \
    || { tail -n 40 "$ROOT/make.log" >&2; exit 1; }
)
"$CHERI_SDK/bin/llvm-strip" --strip-debug -o "$ROOT/perl" "$S/perl"
python3 - "$ROOT" "$HERE" "$CHERI_SDK" "$ARCHIVE" "$CROSS_REV" "$SV_HEADS" <<'PY'
from pathlib import Path
import hashlib, json, re, sys
root, here, sdk, archive = map(Path, sys.argv[1:5])
sv_heads = sys.argv[6] == '1'
source = root / 'src/perl-5.36.3'
config = dict(re.findall(r"^(\w+)='([^']*)'$", (source / 'config.sh').read_text(), re.M))
assert config['ptrsize'] == config['alignbytes'] == '16'
assert config['osname'] == 'freebsd' and config['d_nanosleep'] == 'define'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert ('-DPERL_SV_HEAD_ADAPTER' in config['ccflags']) == sv_heads
manifest = dict(schema=1, application='perl-5.36.3', compiler_target='riscv64-unknown-freebsd13, l64pc128d',
                application_cflags='-O1', pointer_bytes=16, align_bytes=16,
                nested_allocator=('perl-sv-heads: PoisonCap lifetime adapter, PERL_POISONCAP_MODE=0/1'
                                  if sv_heads else 'upstream; no Sublet/PoisonCap inner adapter'),
                source_archive_sha256=sha(archive), perl_cross_revision=sys.argv[5],
                binary_sha256=sha(root / 'perl'), compiler_sha256=sha(sdk / 'bin/clang'),
                config_sha256=sha(source / 'config.sh'),
                inputs_sha256={str(p):sha(p) for p in [here / 'build.sh',
                    *sorted((here / '../musl/patches/5.36.3').glob('*.patch')),
                    *([*sorted((here / '../sv-heads/patches/5.36.3').glob('*.patch')),
                       here / '../sv-heads/sv-heads.h', here / '../sv-heads/cheribsd.c',
                       here / '../../common/include/poisoncap-quarantine-policy.h',
                       here / '../../../experiments/study/reuse-gap-observer.h',
                       here / '../../../experiments/applications/cheribsd-memory.c']
                      if sv_heads else [])]})
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
PY
# A dynamic image builds and installs perfectly and then dies in the loader before
# main, so the link mode is checked here rather than discovered in a guest.
if [[ -n ${PERL_CHERI_STATIC:-} ]]; then
  file "$ROOT/perl" | grep -q 'statically linked' \
    || { echo "PERL_CHERI_STATIC was asked for and the image is not static; the guest loader will refuse it" >&2; exit 2; }
fi
echo "$ROOT/perl"
