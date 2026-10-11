#!/usr/bin/env bash
# Second attempt. The first had two defects, both caught before anything ran:
#
#   1. The _testlimitedcapi Setup line lost its module NAME -- the shell variable
#      started at the first source file -- so makesetup read "_testlimitedcapi.c"
#      as the module and config.c came out without it.
#   2. The binary was stripped with llvm-strip --strip-debug, which this time
#      removed 32 MB. The binary the existing 27 rows were measured with still
#      carries 12 MB of .debug_info, so it was never actually stripped. Stripping
#      now would change two things at once. The new binary is taken unstripped
#      from build/python, so the only difference is the modules.
#
# What .text says about attempt 1: 2,584,846 -> 2,818,472 bytes, so the three
# modules that DID land were really linked in.
set -euo pipefail
T=$CPY_CHERI_BUILD
SL=$T/build/Modules/Setup.local
# No default: a wrong SDK silently produces a binary that runs and reads
# garbage thread-locals (see the -cheri-tgot-tls note in the port's build.sh),
# so these have to be named rather than guessed from one author's layout.
: "${CHERI_SDK:?set CHERI_SDK to the CheriBSD SDK}"
: "${CHERI_SYSROOT:?set CHERI_SYSROOT to the purecap rootfs}"
: "${CPY_CHERI_BUILD:?set CPY_CHERI_BUILD to the existing purecap build tree}"

[[ -f $SL.before ]] || { echo "no $SL.before to restore from" >&2; exit 2; }
cp "$SL.before" "$SL"

LIMITED="_testlimitedcapi _testlimitedcapi.c"          # the module NAME first
for f in abstract bytearray bytes complex dict eval float heaptype_relative import \
         list long object pyos set sys tuple unicode vectorcall_limited file; do
  LIMITED="$LIMITED _testlimitedcapi/$f.c"
done

cat >> "$SL" <<SETUP

# Added 2026-10-08. Five rows on this arm read as the arm staying quiet when the
# trigger had died at import; these are the modules they needed:
#   _interpreters     case 12
#   pyexpat           cases 21 and 22
#   _elementtree      case 22, which reaches its defect through the C parser
#   _testlimitedcapi  case 23
# Flags are taken from the host Makefile rather than guessed:
# MODULE_PYEXPAT_CFLAGS and MODULE__ELEMENTTREE_CFLAGS are both
# -I\$(srcdir)/Modules/expat, and the bundled expat compiles xmlparse.c,
# xmlrole.c and xmltok.c only -- xmltok_impl.c and xmltok_ns.c are #included by
# xmltok.c.
#
# _ctypes is deliberately absent: it needs libffi and the purecap sysroot has
# none, so case 11 stays a declared boundary.
_interpreters _interpretersmodule.c
pyexpat pyexpat.c expat/xmlparse.c expat/xmlrole.c expat/xmltok.c -I\$(srcdir)/Modules/expat
_elementtree _elementtree.c -I\$(srcdir)/Modules/expat
$LIMITED
SETUP

CC="$CHERI_SDK/bin/clang --target=riscv64-unknown-freebsd13 --sysroot=$CHERI_SYSROOT -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -cheri-tgot-tls -fuse-ld=lld -B$CHERI_SDK/bin"
cd "$T/build"
rm -f Modules/config.c Modules/config.o            # force the table to regenerate
start=$(date +%s)
if ! make -j12 python > make-modules2.log 2>&1; then
  echo "FAILED after $(( $(date +%s)-start ))s:"; tail -40 make-modules2.log
  cp "$SL.before" "$SL"; exit 1
fi
echo "make ok in $(( $(date +%s)-start ))s"

# ---- gate: every module asked for must be in the generated table -----------
miss=0
for m in _interpreters pyexpat _elementtree _testlimitedcapi select; do
  if grep -q "PyInit_$m\b" Modules/config.c; then echo "  builtin  $m"
  else echo "  MISSING  $m"; miss=1; fi
done
[[ $miss == 0 ]] || { echo "REFUSING: config.c does not carry every module"; exit 2; }

echo "=== .text, which says whether they were really linked ==="
for f in "$T/python" "$T/build/python"; do
  printf '  %-26s %10s bytes  .text %s\n' "$(basename $(dirname $f))/$(basename $f)" \
    "$(wc -c < $f)" "$($CHERI_SDK/bin/llvm-size -A "$f" | awk '$1==".text"{print $2}')"
done
cp "$T/build/python" "$T/python.modules"
echo "new binary: $T/python.modules  sha $(sha256sum "$T/python.modules" | cut -c1-16)"
