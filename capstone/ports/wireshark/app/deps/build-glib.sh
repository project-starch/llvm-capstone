#!/usr/bin/env bash
# GLib (libglib-2.0 only) for capstone64, cross-configured by GLib's own meson, so config.h's
# answers come from compiling and linking against this libc (capstone-cc), not from a native
# build edited by hand as the M0 census had to.
#
# Only libglib-2.0 is built: tshark links GLib core, not GObject or GIO. meson still requires
# libffi (for GObject) at configure time, so a STUB libffi.pc satisfies the check; nothing
# links against it. pcre2 and zlib are the ones built here (build-pcre2.sh, build-zlib.sh).
#
# Run-time probes meson cannot execute on the build machine are answered in the cross file,
# each from what musl-capstone and the capstone64 ABI do:
#   have_c99_vsnprintf/have_c99_snprintf/have_unix98_printf  musl's printf family (C99, positional)
#   va_val_copy       the capstone64 va_list is a single pointer (a capability): copied by value
#   growing_stack     false: the stack grows down
#   have_strlcpy      musl provides strlcpy
#   have_proc_self_cmdline  false: a domain has no /proc
#
# Gates, as for every library here:
#   1. native: GLib's own glib test suite passes on the host with the port's patches applied
#      (they are __CAPSTONE__-guarded except gqsort's copy, which the qsort tests cover);
#   2. cross: libglib-2.0.a builds with 0 failures and the cast census on, and is installed;
#   3. GLib's own tests for qsort, hash tables, arrays, lists, strings, UTF-8 and GRegex link as
#      capstone64 domains.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$TS_DEPS_DIR/fetch.sh"
TAR=$(ts_fetch glib)
X=$TS_DEPS_BUILD/glib-cap N=$TS_DEPS_BUILD/glib-native LOG=$TS_DEPS_BUILD/glib-logs PC=$TS_DEPS_BUILD/glib-pkgconfig
rm -rf "$X" "$N" "$LOG" "$PC"; mkdir -p "$X" "$N" "$LOG" "$PC"
for d in "$X" "$N"; do
  tar -xf "$TAR" -C "$d" --strip-components=1
  for p in "$TS_DEPS_DIR"/patches/glib-*.patch; do patch -s -d "$d" -p1 < "$p"; done
done

# 1. Native, the glib:glib suite: libglib's own tests (the port builds libglib only; "glib" alone
#    would also select the project's gio suites).
( cd "$N" && env -u CC -u AR -u RANLIB meson setup build -Dtests=true -Dnls=disabled -Dselinux=disabled \
    -Dxattr=false -Dlibmount=disabled -Dman-pages=disabled -Ddocumentation=false -Dintrospection=disabled \
    -Dsysprof=disabled -Dlibelf=disabled -Db_colorout=never > "$LOG/native-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB ninja -C build > "$LOG/native-build.log" 2>&1 &&
  env -u CC -u AR -u RANLIB meson test -C build --suite glib:glib --no-rebuild --num-processes 16 \
    > "$LOG/native-test.log" 2>&1 ) || true
grep -E "^(Ok|Expected Fail|Fail|Unexpected Pass|Skipped|Timeout): " "$LOG/native-test.log" | tr -s ' ' | tr '\n' ' '; echo
nfail=$(awk '/^(Fail|Unexpected Pass|Timeout):/{s+=$2} END{print s+0}' "$LOG/native-test.log")
nok=$(awk '/^Ok:/{print $2}' "$LOG/native-test.log")
[ "${nok:-0}" -gt 50 ] && [ "$nfail" = 0 ] || { echo "glib: NATIVE TEST FAILED (ok ${nok:-0}, fail $nfail)" >&2; exit 1; }
echo "glib: native glib suite OK ($nok ok)"

cp "$TS_DEPS_PREFIX"/lib/pkgconfig/*.pc "$PC/" 2>/dev/null || true
cat > "$PC/libpcre2-8.pc" <<PCEOF
prefix=$TS_DEPS_PREFIX
Name: libpcre2-8
Description: PCRE2 8-bit (built by build-pcre2.sh)
Version: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pcre2"]["version"])' "$TS_DEPS_DIR/deps.json")
Libs: -L\${prefix}/lib -lpcre2-8
Cflags: -I\${prefix}/include -DPCRE2_STATIC
PCEOF
cat > "$PC/libffi.pc" <<PCEOF
Name: libffi
Description: STUB so GLib's meson configures; GObject is not built, and nothing links libffi
Version: 3.4.6
Libs:
Cflags:
PCEOF
cat > "$X/capstone64.cross" <<CROSSEOF
[binaries]
c = '$CC'
ar = '$AR'
pkg-config = 'pkg-config'

[built-in options]
c_args = ['-DCAPSTONE_SINGLE_THREAD_DOMAIN']

[properties]
needs_exe_wrapper = true
pkg_config_libdir = '$PC'
have_c99_vsnprintf = true
have_c99_snprintf = true
have_unix98_printf = true
va_val_copy = true
growing_stack = false
have_strlcpy = true
have_proc_self_cmdline = false

[host_machine]
system = 'linux'
cpu_family = 'riscv64'
cpu = 'riscv64'
endian = 'little'
CROSSEOF
( cd "$X" && PKG_CONFIG_LIBDIR="$PC" meson setup build --cross-file capstone64.cross \
    -Ddefault_library=static -Dtests=false -Dnls=disabled -Dselinux=disabled -Dxattr=false \
    -Dlibmount=disabled -Dman-pages=disabled -Ddocumentation=false -Dintrospection=disabled \
    -Dsysprof=disabled -Dlibelf=disabled -Dglib_debug=disabled -Dinstalled_tests=false \
    -Dbsymbolic_functions=false -Db_colorout=never > "$LOG/cap-configure.log" 2>&1 ) || { tail -20 "$LOG/cap-configure.log"; exit 1; }
echo "glib: configured"
( cd "$X" && TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" ninja -k 0 -C build glib/libglib-2.0.a \
    > "$LOG/cap-build.log" 2>&1 ) || true
grep -E "^FAILED: " "$LOG/cap-build.log" | sed 's/^FAILED: //' | sort > "$LOG/failed.txt" || true
touch "$LOG/cast-log.txt"; sort -u "$LOG/cast-log.txt" > "$LOG/cast-sites.txt"
echo "glib: $(wc -l < "$LOG/failed.txt") objects failed; cast sites: $(wc -l < "$LOG/cast-sites.txt")"
[ -f "$X/build/glib/libglib-2.0.a" ] || { echo "glib: libglib-2.0.a NOT built" >&2; exit 1; }
echo "glib: libglib-2.0.a built"
# No undefined weak symbol: in a domain its address is not NULL (ISSUES C-56), so a `sym != NULL`
# test passes and the call lands on the image base. GLib's LeakSanitizer hooks were two such
# symbols until patch glib-0007; the tshark port's link gate found them.
weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" -A "$X/build/glib/libglib-2.0.a" | awk '$(NF-1) ~ /^[wv]$/ {print $NF}' | sort -u | tr '\n' ' ')
[ -z "$weak" ] || { echo "glib: undefined weak symbols in libglib-2.0.a: $weak" >&2; exit 1; }
echo "glib: no undefined weak symbol in libglib-2.0.a"

# 2. Install: the library, its public headers and the generated glibconfig.h, and a .pc file.
I=$TS_DEPS_PREFIX/include/glib-2.0
rm -rf "$I" "$TS_DEPS_PREFIX/lib/glib-2.0"; mkdir -p "$I" "$TS_DEPS_PREFIX/lib/glib-2.0/include" "$TS_DEPS_PREFIX/lib/pkgconfig"
cp "$X/build/glib/libglib-2.0.a" "$TS_DEPS_PREFIX/lib/"
cp "$X/glib/glib.h" "$X/glib/glib-unix.h" "$X/glib/glib-object.h" "$I/" 2>/dev/null || cp "$X/glib/glib.h" "$X/glib/glib-unix.h" "$I/"
mkdir -p "$I/glib/deprecated"
cp "$X"/glib/*.h "$I/glib/"; cp "$X"/glib/deprecated/*.h "$I/glib/deprecated/"
cp "$X"/build/glib/glib-visibility.h "$X"/build/glib/gversionmacros.h "$I/glib/" 2>/dev/null || true
cp "$X/build/glib/glibconfig.h" "$TS_DEPS_PREFIX/lib/glib-2.0/include/"
# gmodule.h and its generated visibility header, without libgmodule: Wireshark includes
# <gmodule.h> unconditionally (wsutil/version_info.c:28, wiretap/busmaster_priv.h:15), and only
# calls g_module_* with plugins on, which this port builds off.
ninja -C "$X/build" gmodule/gmodule-visibility.h > "$LOG/gmodule-visibility.log" 2>&1
mkdir -p "$I/gmodule"
cp "$X/gmodule/gmodule.h" "$I/"; cp "$X/build/gmodule/gmodule-visibility.h" "$I/gmodule/"
cat > "$TS_DEPS_PREFIX/lib/pkgconfig/glib-2.0.pc" <<PCEOF
prefix=$TS_DEPS_PREFIX
Name: GLib
Description: libglib-2.0 for capstone64 (built by build-glib.sh)
Version: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["glib"]["version"])' "$TS_DEPS_DIR/deps.json")
Requires.private: libpcre2-8
Libs: -L\${prefix}/lib -lglib-2.0
Libs.private: -lpcre2-8 -lm
Cflags: -I\${prefix}/include/glib-2.0 -I\${prefix}/lib/glib-2.0/include
PCEOF
echo "glib: installed into $TS_DEPS_PREFIX"

# 3. GLib's own tests link as domains.
GT=(sort hash array-test slist list string strfuncs utf8-misc regex)
nlink=0
for t in "${GT[@]}"; do
  if "$CC" -O1 -I"$I" -I"$TS_DEPS_PREFIX/lib/glib-2.0/include" -I"$TS_DEPS_PREFIX/include" -I"$X/build/glib" -I"$X/glib" -I"$X/build" \
       -DGLIB_DISABLE_DEPRECATION_WARNINGS -o "$LOG/$t.dom" "$X/glib/tests/$t.c" -lglib-2.0 -lpcre2-8 \
       > "$LOG/link-$t.log" 2>&1; then nlink=$((nlink + 1)); else echo "glib: $t did not link (see $LOG/link-$t.log)" >&2; fi
done
echo "glib: $nlink of ${#GT[@]} GLib test programs link as domains"
[ "$nlink" = "${#GT[@]}" ] || exit 1
