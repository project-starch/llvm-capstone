# Environment for building the CHERI vehicle on a host with NO root access.
# Source this before provision-cheri-vehicle.sh.
#
# Everything cheribuild needs that the host lacks is apt-get download'ed and
# dpkg-deb -x'ed into $LOCAL_DEPS; this file points the toolchain at it. That is
# the same no-sudo pattern provision-cheri-vehicle.sh already uses for makefs,
# extended to the host compiler and the QEMU build dependencies.
#
# Supplied here because this host has none of them installed:
#   clang-19 / lld-19 / libclang-rt-19-dev   cheribuild's host compiler + fast linker
#   libtool                                  libtoolize, required by the QEMU build
#   libattr1-dev, libcap-ng-dev              QEMU 9pfs support (pkg-config checks)
#   libslirp-dev, libslirp0                  QEMU user networking (pkg-config check)
#   makefs                                   disk-image assembly
LOCAL_DEPS=${LOCAL_DEPS:-$HOME/cheri-ws/local-deps}

export PATH="$LOCAL_DEPS/usr/bin:$LOCAL_DEPS/usr/sbin:$LOCAL_DEPS/usr/lib/llvm-19/bin:$PATH"

# libtoolize bakes in /usr/share/libtool and /usr/share/aclocal at build time.
# _lt_pkgdatadir is its documented override ("allow the user to override the
# master libtoolize repository"), but it expects the layout build-aux/ libltdl/
# m4/, which is NOT Debian's layout -- Debian uses /usr/share/libtool for
# pkgltdldir and /usr/share/aclocal for the macros, and ships no libltdl tree at
# all. $LOCAL_DEPS/ltdata is a symlink shim onto Debian's layout. Note also that
# build-aux/config.{guess,sub} are RELATIVE symlinks (../../misc/...), so they
# dangle inside a relocated prefix; $LOCAL_DEPS/usr/share/misc links them back
# to the host copies.
export _lt_pkgdatadir="$LOCAL_DEPS/ltdata"
export ACLOCAL_PATH="$LOCAL_DEPS/usr/share/aclocal:${ACLOCAL_PATH:-/usr/share/aclocal}"

# Headers and libraries for the downloaded -dev packages. pkg-config reports
# -I/usr/include for them (prefix=/usr is baked into the .pc files), so CPATH is
# what actually makes them findable from a relocated prefix.
export PKG_CONFIG_PATH="$LOCAL_DEPS/usr/lib/x86_64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPATH="$LOCAL_DEPS/usr/include:$LOCAL_DEPS/usr/include/x86_64-linux-gnu:${CPATH:-}"
export LIBRARY_PATH="$LOCAL_DEPS/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

# RUNTIME, not just build time: libslirp.so.0 exists only in this prefix, so the
# QEMU we build links against it with no rpath and will not start without this.
# Anything that EXECUTES qemu-system-riscv64cheri must source this file too.
export LD_LIBRARY_PATH="$LOCAL_DEPS/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
