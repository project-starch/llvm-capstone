#!/usr/bin/env bash
# Install, WITHOUT ROOT, everything cheribuild needs that this host lacks, into a
# local prefix. Same mechanism provision-cheri-vehicle.sh already uses for
# makefs: `apt-get download` + `dpkg-deb -x`. Idempotent; safe to re-run.
#
# The package list is not guesswork -- it comes from cheribuild's own checks:
#   grep -nE 'check_required_pkg_config\(|check_required_system_tool\(' \
#     $CHERI_WS/cheribuild/pycheribuild/projects/build_qemu.py
# Taking it from there beats discovering one build failure at a time, which is
# how this was first done and cost three restarts of the QEMU step.
#
# Afterwards: . vehicle-env-no-sudo.sh   before provisioning AND before running.
set -uo pipefail
LOCAL_DEPS=${LOCAL_DEPS:-$HOME/cheri-ws/local-deps}
DEBS=${DEBS:-$HOME/cheri-ws/host-dep-debs}
mkdir -p "$LOCAL_DEPS" "$DEBS"

# REFUSE TO RUN WHILE A BUILD IS USING THE TOOLCHAIN. dpkg-deb -x overwrites
# $LOCAL_DEPS/usr/lib/llvm-19/bin/clang in place; doing that while cheribuild is
# executing it kills the build with
#   exec: .../clang: Text file busy
# which looks like a compiler bug and is not one. Learned the hard way: it
# failed the QEMU build at object 405 of 5212.
if pgrep -x cheribuild.py >/dev/null 2>&1 \
   || ps -eo args= | grep -q '[c]heribuild\.py'; then
  echo "REFUSING: a cheribuild run is in flight; it is executing the very clang" >&2
  echo "          this script would overwrite. Wait for it to finish." >&2
  exit 3
fi

command -v apt-get >/dev/null || { echo "need apt-get"; exit 2; }
command -v dpkg-deb >/dev/null || { echo "need dpkg-deb"; exit 2; }

PKGS=(
  # cheribuild's host compiler. The provision script's wrapper loop probes
  # /usr/bin/clang and skips when absent, leaving --clang-path dangling.
  clang-19 libclang-cpp19 libllvm19 libclang1-19 libclang-common-19-dev
  libclang-rt-19-dev
  # host linker: without ld.lld beside the wrappers, cheribuild falls back to
  # BFD, which for LLVM is slow and memory-hungry.
  lld-19
  # QEMU build
  libtool                       # libtoolize
  libslirp-dev libslirp0        # pkg-config slirp  (runtime .so too)
  libattr1-dev                  # pkg-config libattr
  libcap-ng-dev libcap-ng0      # pkg-config libcap-ng
  # disk-image assembly (the provision script also fetches this itself)
  makefs
)

# Download per package, with one retry, and DO NOT abort on a single failure:
# the Debian mirror returned "400 Bad Request" for one package on a re-run even
# though it had downloaded fine minutes earlier. The verification block at the
# bottom is the real gate -- a package that failed to re-download but is already
# extracted is not a problem, and a package that is genuinely missing will fail
# there with a specific message.
echo "[*] downloading into $DEBS"
missed=()
for p in "${PKGS[@]}"; do
  if ( cd "$DEBS" && apt-get download "$p" ) >>"$DEBS/download.log" 2>&1; then
    continue
  fi
  if ( cd "$DEBS" && apt-get download "$p" ) >>"$DEBS/download.log" 2>&1; then
    continue
  fi
  missed+=("$p")
done
[ ${#missed[@]} -eq 0 ] || echo "    note: could not download: ${missed[*]} (ok if already extracted)"

echo "[*] extracting into $LOCAL_DEPS"
shopt -s nullglob
for d in "$DEBS"/*.deb; do dpkg-deb -x "$d" "$LOCAL_DEPS"; done
shopt -u nullglob

# --- fix the two things that break in a RELOCATED prefix --------------------
# (1) pkg-config files bake in prefix=/usr, so they emit -I/usr/include/... and
#     -L/usr/lib/... which do not contain these packages on this host. meson
#     uses pkg-config, not CPATH, for its -I flags, so QEMU failed with
#     "'libslirp.h' file not found" until these were repointed.
PC="$LOCAL_DEPS/usr/lib/x86_64-linux-gnu/pkgconfig"
for f in slirp libattr libcap-ng; do
  [ -f "$PC/$f.pc" ] && sed -i "s|^prefix=/usr$|prefix=$LOCAL_DEPS/usr|" "$PC/$f.pc"
done

# (2) libtool ships build-aux/config.{guess,sub} as RELATIVE symlinks
#     (../../misc/...), which dangle inside a relocated prefix. Point them at
#     the host copies. Also build the layout shim _lt_pkgdatadir expects
#     (build-aux/ libltdl/ m4/), which is NOT Debian's layout: Debian uses
#     /usr/share/libtool for pkgltdldir, /usr/share/aclocal for the macros, and
#     ships no libltdl tree at all.
mkdir -p "$LOCAL_DEPS/usr/share/misc"
for f in config.guess config.sub; do
  [ -f "/usr/share/misc/$f" ] && ln -sfn "/usr/share/misc/$f" "$LOCAL_DEPS/usr/share/misc/$f"
done
mkdir -p "$LOCAL_DEPS/ltdata"
ln -sfn "$LOCAL_DEPS/usr/share/libtool/build-aux" "$LOCAL_DEPS/ltdata/build-aux"
ln -sfn "$LOCAL_DEPS/usr/share/aclocal"           "$LOCAL_DEPS/ltdata/m4"
ln -sfn "$LOCAL_DEPS/usr/share/libtool"           "$LOCAL_DEPS/ltdata/libltdl"

# --- compiler wrappers the provision script will NOT overwrite --------------
# Its loop does `real=$(command -v /usr/bin/$drv) || continue`, so with no
# /usr/bin/clang it leaves whatever is already here. The -Wno-error= flags are
# specific on purpose: QEMU passes -Werror=incompatible-pointer-types-discards-
# qualifiers, and a blanket -Wno-error does not override a specific -Werror=.
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
WRAP="$CHERI_ROOT/host-cc-wrappers"; mkdir -p "$WRAP"
W='-Wno-error -Wno-error=incompatible-pointer-types-discards-qualifiers'
for drv in clang clang++; do
  printf '#!/bin/sh\nexec %s %s "$@" %s\n' \
    "$LOCAL_DEPS/usr/lib/llvm-19/bin/$drv" "$W" "$W" > "$WRAP/$drv"
  chmod +x "$WRAP/$drv"
done
ln -sfn "$LOCAL_DEPS/usr/lib/llvm-19/bin/ld.lld" "$WRAP/ld.lld"
ln -sfn "$LOCAL_DEPS/usr/lib/llvm-19/bin/lld"    "$WRAP/lld"

# --- verify, rather than assume --------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
. "$SCRIPT_DIR/vehicle-env-no-sudo.sh"
fail=0
echo "[*] cheribuild's required programs:"
for t in libtoolize autoreconf aclocal sed bzip2 make ninja cmake; do
  if command -v "$t" >/dev/null; then printf '  ok       %s\n' "$t"
  else printf '  MISSING  %s\n' "$t"; fail=1; fi
done
echo "[*] cheribuild's required pkg-config modules:"
for m in pixman-1 glib-2.0 libattr libcap-ng slirp; do
  v=$(pkg-config --modversion "$m" 2>/dev/null)
  if [ -n "$v" ]; then printf '  ok       %-10s %s\n' "$m" "$v"
  else printf '  MISSING  %s\n' "$m"; fail=1; fi
done
echo "[*] the headers actually compile (pkg-config flags are not enough on their own):"
T=$(mktemp -d "${TMPDIR:-/tmp}/hostdep-XXXX")
printf '#include <libslirp.h>\n#include <cap-ng.h>\n#include <sys/xattr.h>\nint main(void){return 0;}\n' > "$T/p.c"
if "$WRAP/clang" $(pkg-config --cflags slirp libattr libcap-ng 2>/dev/null) \
     "$T/p.c" $(pkg-config --libs slirp libattr libcap-ng 2>/dev/null) -o "$T/p" 2>"$T/err"; then
  echo "  ok       libslirp.h + cap-ng.h + sys/xattr.h compile and link"
else
  echo "  FAILED   see below"; head -5 "$T/err"; fail=1
fi
rm -rf "$T"
echo "[*] libtoolize runs from the relocated prefix:"
if libtoolize --version >/dev/null 2>&1; then echo "  ok       $(libtoolize --version | head -1)"
else echo "  FAILED"; fail=1; fi

[ "$fail" -eq 0 ] && echo "HOST DEPS: READY" || { echo "HOST DEPS: INCOMPLETE"; exit 1; }
