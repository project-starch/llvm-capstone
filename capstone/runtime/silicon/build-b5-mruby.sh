#!/usr/bin/env bash
# B5 (docs/plans/b0-silicon-delegated-runtime.md): mruby 4.0.0-rc2 as a SILICON-ABI delegated application.
#
# mruby's own rake builds every object (core, gems, the generated gem_init.c/mrblib.c, the mruby tool) through a
# wrapper that adds build-b0-hello.sh's exact compile flags (B0_PRINT_CFLAGS), so they are gp-captable, soft-float
# bitcode. Rake's link step is only recorded; build-b0-hello.sh then does the one full-LTO link (I-8) with musl, the
# runtime and the glue (B0_APP_OBJS), and runs its gates: check-repr under the board's encoder and no-fp-gate.py.
# The same rake run builds mruby natively (build/native/bin/mruby), the oracle.
#
# Required: CAPSTONE_LLVM_BIN, B0_MUSL_ARCHIVE (as build-b0-hello.sh). B5_WORK (default $CAPSTONE_TMP_ROOT/b5m) holds
# the pinned mruby tree (fetched and patched here if absent), the wrapper, and OUT_DIR (default $B5_WORK/out).
# Board encoder: B0_CAP_FORMAT=full B0_CAP_RENORM=1 with the pool carve (exact under the full scheme, confirmed on
# silicon by boot W) are the defaults here, for caplifive_fpb_afcd19bea.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAP=$(cd "$HERE/../.." && pwd)
BIN=${CAPSTONE_LLVM_BIN:?set CAPSTONE_LLVM_BIN}
: "${B0_MUSL_ARCHIVE:?set B0_MUSL_ARCHIVE}"
W=${B5_WORK:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/b5m}
M=$W/src/mruby
PATCHES=$CAP/ports/mruby/app/patches/4.0.0-rc2
MRUBY_COMMIT=9d523e2f74f2e63ca02840937523de61398a617d
mkdir -p "$W/bin"

# ---- mruby at the 4.0.0-rc2 pin, with the port's patches (as build-mruby-domain.sh) ------------------------------
PATCH_HASH=$(sha256sum "$PATCHES"/*.patch | sha256sum | cut -d' ' -f1)
if [ ! -d "$M/.git" ]; then
  mkdir -p "$M"
  git -C "$M" init -q
  git -C "$M" remote add origin https://github.com/mruby/mruby.git
  git -C "$M" fetch -q --depth 1 origin "$MRUBY_COMMIT"
  git -C "$M" checkout -q FETCH_HEAD
  for p in "$PATCHES"/*.patch; do (cd "$M" && patch -p1 -s < "$p") || { echo "patch $p did not apply" >&2; exit 2; }; done
  echo "$PATCH_HASH" > "$M/.b5-patchset"
fi
[ "$(git -C "$M" rev-parse HEAD)" = "$MRUBY_COMMIT" ] || { echo "$M is not at $MRUBY_COMMIT" >&2; exit 2; }
[ "$(cat "$M/.b5-patchset" 2>/dev/null)" = "$PATCH_HASH" ] || { echo "$M was patched with another patch set" >&2; exit 2; }

# ---- the wrapper: build-b0-hello.sh's flags, verbatim -------------------------------------------------------------
B0_PRINT_CFLAGS=1 bash "$HERE/build-b0-hello.sh" > "$W/silicon-cflags.txt"
[ "$(wc -l < "$W/silicon-cflags.txt")" -gt 10 ] || { echo "B0_PRINT_CFLAGS printed nothing usable" >&2; exit 2; }
cat > "$W/bin/silicon-cc" <<EOF
#!/usr/bin/env bash
mapfile -t F < "$W/silicon-cflags.txt"
exec "$BIN/clang" "\${F[@]}" "\$@"
EOF
cat > "$W/bin/silicon-ld-record" <<'EOF'
#!/usr/bin/env bash
# Rake's link step for the silicon build: record the inputs, make an empty output, link nothing.
out=; prev=
for a in "$@"; do [ "$prev" = "-o" ] && out=$a; prev=$a; done
[ -n "$out" ] || { echo "silicon-ld-record: no -o" >&2; exit 1; }
printf '%s\n' "$@" > "$out.link-args"
: > "$out"
EOF
chmod +x "$W/bin/silicon-cc" "$W/bin/silicon-ld-record"

# ---- rake: native (the oracle, and mrbc) and silicon ---------------------------------------------------------------
rm -f "$M/build/silicon/bin/mruby" "$M/build/silicon/bin/mruby.link-args"
(cd "$M" && MRUBY_CONFIG="$CAP/ports/mruby/app/build_config-silicon.rb" SILICON_CC="$W/bin/silicon-cc" \
  SILICON_LD_RECORDER="$W/bin/silicon-ld-record" LLVM_AR="$BIN/llvm-ar" rake -j"${B5_JOBS:-8}" all) \
  > "$W/rake.log" 2>&1 || { tail -15 "$W/rake.log" >&2; echo "rake failed (see $W/rake.log)" >&2; exit 2; }
ARGS=$M/build/silicon/bin/mruby.link-args
[ -f "$ARGS" ] || { echo "rake recorded no link for the silicon mruby" >&2; exit 2; }
OBJS=$(python3 - "$ARGS" "$M" <<'PY'
import sys, os
args = open(sys.argv[1]).read().split("\n")
root = sys.argv[2]
out = [a for a in args if a.endswith((".o", ".a")) and not a.startswith("-")]
print(" ".join(os.path.join(root, a) if not os.path.isabs(a) else a for a in out))
PY
)
[ -n "$OBJS" ] || { echo "no objects in $ARGS" >&2; exit 2; }
# patch 0009's delegated IO.popen calls capstone_mruby_spawn, which the port keeps in spawn-shell.c (the virtual
# build links it as MRBD_SPAWN_OBJECT): compiled here with the same silicon flags.
"$W/bin/silicon-cc" -c "$CAP/ports/mruby/app/spawn-shell.c" -o "$W/spawn-shell.o"
# silicon-alloc.c: mruby's allocator (mrb_basic_alloc_func) on size classes, since level0's first-fit list is
# O(live blocks) per call. It goes FIRST: lld then never extracts allocf.o from libmruby.a, whose definition would
# otherwise collide with it (a duplicate-symbol error, not a silent choice).
"$W/bin/silicon-cc" -c "$CAP/ports/mruby/app/silicon-alloc.c" -o "$W/silicon-alloc.o" -DB5_SLAB_BYTES=${B5_SLAB:-$((16 << 20))}
OBJS="$W/silicon-alloc.o $OBJS $W/spawn-shell.o"
echo "silicon mruby: $(echo $OBJS | wc -w) objects/archives from rake; native $(ls -la "$M/build/native/bin/mruby" | awk '{print $5}') bytes"

export B0_APP=mruby B0_APP_OBJS="$OBJS"
export B0_STACK=${B0_STACK:-$((1 << 20))} B0_DATA=${B0_DATA:-$((40 << 20))} B0_ARENA=${B0_ARENA:-$((16 << 20))}
export B0_CAP_FORMAT=${B0_CAP_FORMAT:-full} B0_CAP_RENORM=${B0_CAP_RENORM-1} B0_GLUE_CARVE=${B0_GLUE_CARVE:-pool}
export OUT_DIR=${OUT_DIR:-$W/out}
exec bash "$HERE/build-b0-hello.sh"
