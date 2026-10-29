#!/usr/bin/env bash
# PHP 5.0.0, cross-built for CheriBSD purecap: the CHERI baseline for the two php bug corpora.
#
# WHAT THIS IS FOR. capstone/tests/cheri-php-blind measures CHERI against ROWS that model the
# allocator's mechanism, because "running the real triggers needs a full PHP 5.0.0 purecap build,
# which is gated separately" (that README's Scope section). This script is that gate: it produces
# the real interpreter, so the corpora's cheribsd-revocation arm runs the same trigger scripts the
# Capstone arms run.
#
# THE SAME SOURCE AS THE OTHER ARMS. The tree patches are app/patches 0001 and 0002, the two the
# virtual port also applies; 0003 is the Capstone seam and is deliberately NOT applied, because
# this arm is stock CheriBSD. prepare-sources.sh is not reused: it configures natively for the
# virtual build's translation-unit list, and this build needs its own cross configure.
#
# FOUR CROSS ANSWERS AND TWO CONFIGURE PATCHES, each because PHP 5.0.0's configure can only settle
# the question by RUNNING a program, which a cross build cannot do. Every value below is a fact
# about the target that is checkable without running anything -- from CheriBSD's own headers or
# from the ABI -- and none of them is a guess about PHP.
#
# PHP_CHERI_CONFIGURE_EXTRA appends configure arguments, so a second binary can carry the
# bundled extensions four corpus cases need (mbstring, pcre) without changing the default one:
# the 52 cases that do not need them keep the --disable-all build they were measured on.
# PHP_CHERI_EXTRA_CFLAGS appends CFLAGS for the same reason. The extensions build needs
# -DHAVE_STDARG_PROTOTYPES=1: bundled oniguruma still carries the pre-C89 varargs spelling
# (va_dcl and <varargs.h>) behind that guard, and clang 17 rejects it outright. The macro is
# not a guess about the target -- every compiler that can build this tree has stdarg
# prototypes; it is a configure answer PHP 5.0.0 derives by compiling a program.
#
#   usage: CHERI_SDK=... CHERI_SYSROOT=... build.sh [out-dir]
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SDK=${CHERI_SDK:-$HOME/cheriBSD/cheri/output/sdk}
SYSROOT=${CHERI_SYSROOT:-$HOME/cheriBSD/cheri/output/rootfs-riscv64-purecap}
OUT=${1:-${PHP_CHERI_ROOT:-/tmp/capstone/php-cheribsd}}
VERSION=5.0.0
SHA256=5783e0c0ba94f165633a565fe73a83e59cf17b6880ef95fa8f936dc6301d6919
URL=https://museum.php.net/php5/php-${VERSION}.tar.gz
TARBALL=${PHP_ARCHIVE:-$OUT/cache/php-${VERSION}.tar.gz}
T=$OUT/php-${VERSION}

for p in "$SDK/bin/clang" "$SYSROOT/usr/include/stdio.h"; do
  [ -e "$p" ] || { echo "MISSING: $p (set CHERI_SDK / CHERI_SYSROOT)" >&2; exit 2; }
done
mkdir -p "$OUT/cache"
if [ ! -f "$TARBALL" ]; then
  curl -fL "$URL" -o "$TARBALL"
fi
echo "$SHA256  $TARBALL" | sha256sum -c - >/dev/null \
  || { echo "php-${VERSION}.tar.gz does not match the pinned SHA-256" >&2; exit 2; }

rm -rf "$T"
tar -xzf "$TARBALL" -C "$OUT"
for p in "$HERE"/../app/patches/php-${VERSION}-0001-*.patch "$HERE"/../app/patches/php-${VERSION}-0002-*.patch; do
  patch -d "$T" -p1 -s < "$p" || { echo "tree patch failed: $p" >&2; exit 2; }
  echo "  applied $(basename "$p")"
done

# ---- the four answers a cross build cannot measure -----------------------------------------
# fopencookie:  CheriBSD declares it (usr/include/stdio.h:436). PHP's own probe would have to run.
# funopen:      ALSO declared -- and PHP 5.0.0 checks funopen FIRST and, under HAVE_FUNOPEN,
#               defines its own 2004 `fopencookie` wrapper whose cookie struct has reader/writer/
#               seeker/closer. Against a libc that now declares the real one that is a type and a
#               prototype conflict (7 errors in main/streams/cast.c). Answering `no` here is not a
#               claim that funopen is absent: it is the only way to tell a 2004 configure to take
#               the native route, and cast.c is the only file in the build that reads the answer.
# getaddrinfo:  present and working in CheriBSD's libc; the probe RESOLVES a name to decide.
# time_r type:  FreeBSD's localtime_r/gmtime_r are the POSIX ones.
cat > "$T/config.cache" <<'EOF'
ac_cv_func_fopencookie=${ac_cv_func_fopencookie='yes'}
ac_cv_func_funopen=${ac_cv_func_funopen='no'}
ac_cv_func_getaddrinfo=${ac_cv_func_getaddrinfo='yes'}
ac_cv_time_r_type=${ac_cv_time_r_type='POSIX'}
EOF

# ---- the two probes that have no cache variable at all --------------------------------------
# They write a file from a RUN, so there is nothing to seed; the cross arm of each is patched to
# the target's own answer instead.
python3 - "$T" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "configure"
lines = p.read_text(errors="replace").splitlines(True)
BAIL = "can not run test program while cross compiling"
done = []
for i, line in enumerate(lines):
    if BAIL not in line:
        continue
    before = "".join(lines[max(0, i - 8):i])
    if "MM alignment" in before:
        lines[i] = (
            "    # CROSS: the probe runs a program to read __alignof__ of a union of void*, long\n"
            "    # and double. A purecap pointer is a 128-bit capability aligned to 16, so the\n"
            "    # union's alignment is 16 and its log2 is 4. ZEND_MM is not compiled at 5.0.0\n"
            "    # (Zend/zend_mm.h: /* #define ZEND_MM */), so these only land as defines.\n"
            "    LIBZEND_MM_ALIGN=${LIBZEND_MM_ALIGN:-16}; LIBZEND_MM_ALIGN_LOG2=${LIBZEND_MM_ALIGN_LOG2:-4}\n"
            "    cat >> confdefs.h <<EOF\n#define ZEND_MM_ALIGNMENT $LIBZEND_MM_ALIGN\nEOF\n"
            "    cat >> confdefs.h <<EOF\n#define ZEND_MM_ALIGNMENT_LOG2 $LIBZEND_MM_ALIGN_LOG2\nEOF\n")
        done.append("MM alignment")
    elif "have_fopen_cookie=yes" in before:
        lines[i] = (
            "    # CROSS: the probe runs a program to see whether the cookie seeker takes off64_t.\n"
            "    # CheriBSD's stdio.h answers it statically: cookie_seek_function_t is\n"
            "    # int (*)(void *, off64_t *, int) (usr/include/stdio.h:428), so it does.\n"
            "    cookie_io_functions_use_off64_t=yes\n")
        done.append("cookie off64_t")
if len(done) != 2:
    sys.exit(f"configure: expected to patch 2 cross probes, patched {done}")
p.write_text("".join(lines))
print("  pre-answered:", ", ".join(done))
PY

# ---- configure ------------------------------------------------------------------------------
# A CONFIGURE-ONLY lex stub, for the same reason the virtual build needs one on a host without
# flex: PHP 5.0.0 ships its generated scanners and the timestamp guard below keeps make from
# regenerating them, but autoconf's AC_PROG_LEX probes for lex regardless.
if ! command -v flex >/dev/null 2>&1; then
  mkdir -p "$OUT/shim"
  printf '#!/bin/sh\nprintf "char *yytext;\\nint yywrap(void) { return 1; }\\n" > lex.yy.c\nexit 0\n' \
    > "$OUT/shim/flex"
  cp "$OUT/shim/flex" "$OUT/shim/lex"
  chmod +x "$OUT/shim/flex" "$OUT/shim/lex"
  PATH="$OUT/shim:$PATH"
  echo "  flex not on PATH: using a configure-only stub (no generated file is rebuilt)"
fi
for f in config.guess config.sub; do
  src=$(ls /usr/share/misc/$f /usr/share/automake-*/$f 2>/dev/null | head -1) || true
  [ -n "${src:-}" ] && cp -f "$src" "$T/$f" && chmod +x "$T/$f"
done

# -cheri-tgot-tls: the SDK's own purecap configuration passes it; without it the purecap loader
# refuses a dynamic image before main with "Traditional TLS not supported" (measured for the Perl
# port, 2026-10-06). CPP carries NO linker flag: autoconf 2.13 reads any stderr output from the
# preprocessor as a failed header check, so one clang warning about an unused -fuse-ld makes every
# <netinet/in.h>, <pwd.h>, <locale.h> check answer "no" and the build then fails in five files.
# PHP_CHERI_HYBRID=1 builds the SAME source for the SAME guest and the SAME allocator with the
# capability ABI OFF. It exists to isolate one variable: when a case ends differently here than
# under purecap, the difference is CHERI's and not the operating system's or jemalloc's. The
# hybrid world lives in the purecap rootfs's lib64 tree, with its own rtld.
if [ "${PHP_CHERI_HYBRID:-0}" = 1 ]; then
  TARGET="--target=riscv64-unknown-freebsd13 --sysroot=$SYSROOT -march=rv64gc -mabi=lp64d -mno-relax"
  LINKFLAGS="-L$SYSROOT/usr/lib64 -B$SYSROOT/usr/lib64 -Wl,-dynamic-linker,/libexec/ld-elf64.so.1"
  WANT_ABI=hybrid
else
  TARGET="--target=riscv64-unknown-freebsd13 --sysroot=$SYSROOT -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -cheri-tgot-tls"
  LINKFLAGS=""
  WANT_ABI=purecap
fi
cd "$T"
CC="$SDK/bin/clang $TARGET $LINKFLAGS -fuse-ld=lld -B$SDK/bin -Qunused-arguments" \
CPP="$SDK/bin/clang $TARGET -E" \
CFLAGS="-O0 -g -std=gnu89 -fcommon -Wno-everything ${PHP_CHERI_EXTRA_CFLAGS:-}" \
  ./configure --cache-file=./config.cache \
    --host=riscv64-unknown-freebsd --build="$(./config.guess 2>/dev/null || echo x86_64-pc-linux-gnu)" \
    --disable-all --enable-cli --disable-cgi --disable-debug ${PHP_CHERI_CONFIGURE_EXTRA:-} \
    > "$OUT/configure.log" 2>&1 \
  || { echo "configure FAILED -- tail:"; tail -20 "$OUT/configure.log"; exit 1; }

# The shipped parser/scanner stay shipped: .y/.l older than the .c they produced.
for g in Zend/zend_language_parser.y Zend/zend_language_scanner.l Zend/zend_ini_parser.y \
         Zend/zend_ini_scanner.l ext/standard/parsedate.y; do
  [ -f "$g" ] && touch -d '2004-01-01' "$g"
done
for o in Zend/zend_language_parser.c Zend/zend_language_parser.h Zend/zend_language_scanner.c \
         Zend/zend_ini_parser.c Zend/zend_ini_parser.h Zend/zend_ini_scanner.c \
         ext/standard/parsedate.c; do
  [ -f "$o" ] && touch -d '2005-01-01' "$o"
done

make -j"${JOBS:-16}" > "$OUT/make.log" 2>&1 \
  || { echo "make FAILED -- tail:"; tail -30 "$OUT/make.log"; exit 1; }
[ -x sapi/cli/php ] || { echo "no sapi/cli/php produced" >&2; exit 1; }

# A BUILD THAT IS NOT PURECAP WOULD BUILD AND RUN, AND MEASURE NOTHING. The ABI is asserted.
flags=$("$SDK/bin/llvm-readelf" -h sapi/cli/php | sed -n 's/^  Flags: *//p')
case "$WANT_ABI:$flags" in
  purecap:*cheriabi*capability\ mode*) ;;
  hybrid:*cheriabi*) echo "asked for hybrid, built purecap: flags = $flags" >&2; exit 1;;
  hybrid:*) ;;
  *) echo "sapi/cli/php is not purecap: flags = $flags" >&2; exit 1;;
esac
echo "--> built $T/sapi/cli/php"
echo "    flags: $flags"
echo "    sha256: $(sha256sum sapi/cli/php | cut -d' ' -f1)"
