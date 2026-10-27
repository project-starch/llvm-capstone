#!/bin/bash
# Slot census of PARSEC 3.0 (the 13 C and C++ packages) with its simlarge and native inputs, through PARSEC's own
# parsecmgmt. The tracer is attached by parsecmgmt's submit hook (-s), so only the benchmark binary runs under it.
#
# usage: run-parsec.sh OUT_DIR [INPUT...]   (default simlarge native)
# Uses parsec-3.0-core.tar.gz and the input archives from $SLOTCENSUS_CACHE (default ~/slot-census/cache; the
# Princeton site is gone, the files come from its Internet Archive copy) and records their sha256. Configuration gcc
# (swaptions: gcc-pthreads, see below) with $THREADS threads. GCC 15 rejects much of PARSEC's 2009 code as written, so
# the configuration's flags gain -std=gnu11 for C, -std=gnu++11 for C++, and the warnings GCC 14 made errors are
# demoted back to warnings; bodytrack and facesim get -std=gnu++98, the bundled OpenSSL a termio.h compatibility
# header and a no-op pod2man, x264 -no-pie, and HUGE glibc's former definition (see below). Nothing else in PARSEC is
# changed. A package that does not build is recorded as such. A cell is valid when the benchmark exits 0 (PARSEC has
# no output checks).
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
INPUTS=("$@"); [ ${#INPUTS[@]} -gt 0 ] || INPUTS=(simlarge native)
THREADS=${SLOTCENSUS_PARSEC_THREADS:-8}
PKGS=(blackscholes bodytrack canneal dedup facesim ferret fluidanimate freqmine raytrace streamcluster swaptions vips
      x264)
. "$HERE/tracer.sh"
start_pass

ARCH=(parsec-3.0-core.tar.gz)
for i in "${INPUTS[@]}"; do
  case $i in
    simsmall|simmedium|simlarge) ARCH+=(parsec-3.0-input-sim.tar.gz);;
    native)   # the Internet Archive's copy ends after 1,014,297,656 of 2,429,601,226 bytes (two downloads stopped at
              # that byte): its truncated form, under that name, holds the native inputs stored before the cut intact
      if [ -s "$CACHE/parsec-3.0-input-native.tar.gz" ]; then ARCH+=(parsec-3.0-input-native.tar.gz)
      else ARCH+=(parsec-3.0-input-native.truncated.tar.gz); fi;;
    *) echo "run-parsec: unknown input $i" >&2; exit 2;;
  esac
done
mapfile -t ARCH < <(printf '%s\n' "${ARCH[@]}" | awk '!seen[$0]++')
for a in "${ARCH[@]}"; do
  [ -s "$CACHE/$a" ] || { echo "run-parsec: $CACHE/$a missing" >&2; exit 1; }
  if [ "$a" = parsec-3.0-input-native.truncated.tar.gz ]; then
    tar xzf "$CACHE/$a" -C "$TOOL" 2> "$OUT/logs/native-extract.log"   # ends in an unexpected EOF, by construction
  else
    tar xzf "$CACHE/$a" -C "$TOOL" || exit 1
  fi
done
P=$TOOL/parsec-3.0
C=$P/config/gcc.bldconf
DEMOTE="-Wno-error=implicit-function-declaration -Wno-error=int-conversion -Wno-error=incompatible-pointer-types"
DEMOTE="$DEMOTE -Wno-error=implicit-int"
# glibc 2.43 no longer defines the SVID macro HUGE (ferret uses it); give it glibc's former value, FLT_MAX
DEMOTE="$DEMOTE -DHUGE=3.40282347e+38F"
sed -i -e "s|^export CFLAGS=\" -O3 |export CFLAGS=\" -O3 -std=gnu11 -fcommon $DEMOTE |" \
       -e 's|^export CXXFLAGS="-O3 |export CXXFLAGS="-O3 -std=gnu++11 |' "$C"
grep -q 'std=gnu11' "$C" && grep -q 'std=gnu++11' "$C" || { echo "run-parsec: gcc.bldconf changed" >&2; exit 1; }
# bodytrack and facesim are C++98: their destructors carry throw() lists and facesim compares a stream with 0.
# A package's own build configuration is read after the global one, so its -std comes last on the command line.
for pkg in bodytrack facesim; do
  echo 'CXXFLAGS="${CXXFLAGS} -std=gnu++98"' >> "$P/pkgs/apps/$pkg/parsec/gcc.bldconf"
done
# dedup's bundled OpenSSL 0.9.8 includes termio.h on linux-x86_64 (its target sets -DTERMIO), and glibc 2.43 no
# longer ships termio.h. A compatibility header gives it what glibc's own header did, termios.h and sys/ioctl.h,
# with struct termio spelled as struct termios. The code that uses it is OpenSSL's terminal password prompt, which
# dedup (it uses this OpenSSL for SHA-1) never calls.
mkdir -p "$TOOL/compat"
printf '#include <termios.h>\n#include <sys/ioctl.h>\n#define termio termios\n' > "$TOOL/compat/termio.h"
SSL=$P/pkgs/libs/ssl/parsec/gcc.bldconf
sed -i "s|^build_conf=\"threads no-zlib no-shared\"$|build_conf=\"threads no-zlib no-shared -I$TOOL/compat\"|" "$SSL"
grep -q -- "-I$TOOL/compat\"$" "$SSL" || { echo "run-parsec: ssl gcc.bldconf changed" >&2; exit 1; }
# its "make install" also builds the man pages, whose 2009 POD today's pod2man rejects. A pod2man first on the PATH
# of the OpenSSL build only writes a one-word placeholder page; OpenSSL's util/pod2mantest accepts a pod2man whose
# output names OpenSSL, so it is chosen over the system one. Only the documentation changes.
mkdir -p "$TOOL/compat/bin"
printf '#!/bin/sh\necho OpenSSL\n' > "$TOOL/compat/bin/pod2man" && chmod 755 "$TOOL/compat/bin/pod2man"
sed -i "s|^build_env=\"PATH=\\\\\"\${CC_HOME}/bin:|build_env=\"PATH=\\\\\"$TOOL/compat/bin:\${CC_HOME}/bin:|" "$SSL"
grep -q "$TOOL/compat/bin:" "$SSL" || { echo "run-parsec: ssl build_env changed" >&2; exit 1; }
# x264's hand-written assembly is not position independent and GCC 15 links PIE by default (2009 GCC did not)
echo 'LDFLAGS="${LDFLAGS} -no-pie"' >> "$P/pkgs/apps/x264/parsec/gcc.bldconf"
# facesim (PhysBAM) crashes at start-up at -O1, -O2 and -O3 under GCC 15 and runs at -O0; -fno-lifetime-dse,
# -fno-strict-aliasing and -fno-delete-null-pointer-checks each still crash. It is built at -O0, the only level at
# which its 2009 code runs; its allocations are program logic, which the optimisation level does not change.
echo 'CXXFLAGS="${CXXFLAGS} -O0"' >> "$P/pkgs/apps/facesim/parsec/gcc.bldconf"
# canneal's thread entry point is declared void* and falls off its end, which GCC 15 at -O3 compiles as unreachable:
# the thread runs on into the next function and overflows its stack. The one source change in PARSEC: return NULL
# there (the value is never read, as pthread_join is called with NULL).
CM=$P/pkgs/kernels/canneal/src/main.cpp
RE='s/(void\* entry_pt\(void\* data\)\n\{\n\tannealer_thread\* ptr = static_cast<annealer_thread\*>\(data\);\n'
RE+='\tptr->Run\(\);\n)\}/$1\treturn NULL;\n}/'
perl -0pi -e "$RE" "$CM"
grep -q "return NULL;" "$CM" || { echo "run-parsec: canneal main.cpp changed" >&2; exit 1; }

# Instruction passes do not follow execs: the benchmark is one process, and facesim's system("mkdir -p ...") under
# DynamoRIO started the multicall coreutils as "libdynamorio" ("coreutils: unknown program", facesim exit 255)
if [ "$MEASURE" = instructions ]; then LAUNCH="$DRRUN -no_follow_children -c $SLOTINS \$CENSUS_FILE --"
else LAUNCH="env SLOTCENSUS_OUT=\$CENSUS_FILE LD_PRELOAD=$TOOL/slotcensus.so"; fi
cat > "$TOOL/submit.sh" <<EOF
#!/bin/bash
$LAUNCH "\$@"
echo \$? > "\$CENSUS_RC"
EOF
chmod 755 "$TOOL/submit.sh"
{
  provenance_tracer
  (cd "$CACHE" && sha256sum "${ARCH[@]}")
  echo "parsec $(cat "$P/version") config gcc, threads $THREADS, inputs ${INPUTS[*]}"
  grep -E '^export (CFLAGS|CXXFLAGS)=' "$C"
} > "$OUT/provenance.txt"

declare -A EXE=([raytrace]=rtview)
# PARSEC's gcc configuration builds swaptions' TBB variant, whose bundled TBB does not build here; its gcc-pthreads
# configuration builds the same benchmark on pthreads, as every other package runs
declare -A CONF=([swaptions]=gcc-pthreads)
cd "$P" || exit 1
for pkg in "${PKGS[@]}"; do
  ( set +u; . ./env.sh && parsecmgmt -a build -p "$pkg" -c "${CONF[$pkg]:-gcc}" ) > "$OUT/logs/build-$pkg.log" 2>&1
  # built means the installed binary exists (the logs echo -Wno-error flags, so grepping them for errors misleads)
  if compgen -G "$P/pkgs/*/$pkg/inst/*.${CONF[$pkg]:-gcc}/bin/${EXE[$pkg]:-$pkg}" > /dev/null; then
    echo "build $pkg ok" >> "$OUT/provenance.txt"
  else
    echo "build $pkg FAILED" >> "$OUT/provenance.txt"
  fi
done
: > "$OUT/cells.tsv"
for input in "${INPUTS[@]}"; do
  for pkg in "${PKGS[@]}"; do
    cell=parsec-$input-$pkg
    rm -f "$OUT/$cell.rc"
    # a package that reads an input file and has no intact one for this input set (the truncated native archive cuts
    # x264's and holds none after it) is recorded as such, without a run
    indir=$(compgen -G "$P/pkgs/*/$pkg/inputs" | head -1)
    if [ -n "$indir" ] && ! tar tf "$indir/input_$input.tar" > /dev/null 2>&1; then   # absent or cut short
      record_cell "$cell" "${EXE[$pkg]:-$pkg}" INVALID input-missing
      continue
    fi
    t=$(date +%s)
    ( set +u; . ./env.sh && CENSUS_FILE=$OUT/$cell.jsonl CENSUS_RC=$OUT/$cell.rc \
        parsecmgmt -a run -p "$pkg" -c "${CONF[$pkg]:-gcc}" -i "$input" -n "$THREADS" -s "$TOOL/submit.sh" ) \
      > "$OUT/logs/$cell.log" 2>&1
    rc=$(cat "$OUT/$cell.rc" 2>/dev/null || echo none)
    v=INVALID; [ "$rc" = 0 ] && v=valid
    record_cell "$cell" "${EXE[$pkg]:-$pkg}" "$v" "rc=$rc $(took $t)"
    rm -rf "$P"/pkgs/*/"$pkg"/run   # the unpacked input and outputs; native inputs reach gigabytes
  done
done
finish_pass
