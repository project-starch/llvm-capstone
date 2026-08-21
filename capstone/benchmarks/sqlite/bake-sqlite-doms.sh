#!/usr/bin/env bash
# Build several SQLite domain variants and bake them into ONE firmware image, then PROVE each
# one is in the image by hash.
#
#   bake-sqlite-doms.sh sqwho:CAPSTONE_DBWHO_PROBE=1 \
#                       sqfix:CAPSTONE_DBWHO_PROBE=1,CAPSTONE_DBFIX_PROBE=1
#
# Each argument is <name>:<VAR=VAL>[,<VAR=VAL>...]. The domain is baked as <name>.dom.
#
# WHY A SCRIPT. Three separate things here have each cost a board session when done by hand:
#
#  * ONE FIRMWARE REBUILD COVERS EVERY DOMAIN, so the marginal cost of the fourth variant is a
#    few seconds. Sending variants one per boot is how ~5 minutes per hypothesis became ~5
#    minutes per question. Build them all, then rebuild once.
#
#  * `A=linux-rebuild` MUST RUN BEFORE `A=opensbi-rebuild`. Buildroot does not track
#    overlay/ -> cpio, so an OpenSBI-only relink ships the PREVIOUS initramfs while reporting
#    success, and the board then runs yesterday's domain under today's name.
#
#  * STAGING IS VERIFIED BY CONTENT HASH EXTRACTED FROM THE CPIO, never by filename or mtime.
#    A killed bake once left the right name in the overlay and the wrong bytes in the image;
#    every check that looks at the filesystem passed. The cpio is the only thing the board
#    actually receives, so it is the only thing worth checking.
#
# BAKE_EXTRA_FILES="a,b,c" additionally stages arbitrary files by path -- the SQLLogicTest
# .test corpus slices and the sqlite_host.user binary, which are data rather than domains and
# so have no <name>:<VAR=VAL> build of their own. They are hash-verified inside the cpio
# exactly like a domain, and pruned from their OWN manifest by exact name, never by glob:
# this script still removes only what it can prove it created.
#
# Exits non-zero on any mismatch. A missing artifact is an ERROR, never a quiet skip: a bake
# that silently staged nothing looks exactly like a bake that staged everything.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"
ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

SYS=${CAPLIFIVE_SYSTEM:-$ROOT/capstone/caplifive-system}
BR="$SYS/sw/buildroot"
OVERLAY="$BR/overlay/test-domains"
TARGET="$BR/build/target/test-domains"
CPIO="$BR/build/images/rootfs.cpio"

[[ $# -gt 0 ]] || { echo "usage: $0 <name>:<VAR=VAL>[,...] ..." >&2; exit 2; }
[[ -d "$OVERLAY" && -d "$TARGET" ]] || { echo "ERROR: overlay/target missing under $BR" >&2; exit 1; }

# THE IMAGE IS PRUNED EVERY BAKE, from a manifest of what THIS script staged last time.
#
# Nothing removed the previous set, so every bake added ~1.5 MB per variant and kept the last
# one: fourteen SQLite domains, 23 MB of them, of which four were live. That is pure cost on the
# JTAG upload of every subsequent boot, and it grows monotonically over a debugging session --
# exactly when boots are most frequent.
#
# BY EXPLICIT NAME, FROM A MANIFEST, NEVER A GLOB. A prefix glob once deleted the
# package-installed sbi.dom, so the rule here is that this script removes only what it can prove
# it created. Domains it never staged -- k800, the s06agg set, lpc, sqlite_host.user -- are
# invisible to it. BAKE_KEEP="a,b" spares names from the previous set that are still wanted.
MANIFEST="$BR/.sqlite-bake-manifest"
_img_size() { stat -c%s "$BR/build/images/rootfs.cpio" 2>/dev/null || echo 0; }
SIZE_BEFORE=$(_img_size)

declare -a NAMES=() WANT=()

for spec in "$@"; do
  name=${spec%%:*}
  envs=${spec#*:}
  [[ "$name" != "$spec" ]] || { echo "ERROR: '$spec' has no ':<VAR=VAL>' part" >&2; exit 2; }
  out="$CAPSTONE_TMP_ROOT/bake/$name"
  rm -rf "$out"
  echo "== building $name  [$envs]"
  # The env assignments are applied to THIS build only, so two variants cannot leak flags into
  # each other -- the failure mode that made an "unfixed" control quietly carry the fix.
  ( set -a; IFS=','; for kv in $envs; do export "${kv?}"; done; set +a
    OUT_DIR="$out" bash "$SCRIPT_DIR/build-sqlite-silicon.sh" ) >"$CAPSTONE_TMP_ROOT/bake-$name.log" 2>&1 \
    || { echo "ERROR: build of $name failed; see $CAPSTONE_TMP_ROOT/bake-$name.log" >&2
         tail -25 "$CAPSTONE_TMP_ROOT/bake-$name.log" >&2; exit 1; }
  src="$out/sqlite_silicon.dom"
  [[ -f "$src" ]] || { echo "ERROR: $name built but $src does not exist" >&2; exit 1; }
  h=$(sha256sum "$src" | cut -d' ' -f1)
  cp -f "$src" "$OVERLAY/$name.dom"
  cp -f "$src" "$TARGET/$name.dom"
  NAMES+=("$name"); WANT+=("$h")
  echo "   $name.dom  $h  ($(stat -c%s "$src") bytes)"
done

# EXTRA FILES -- data, staged and verified with the same discipline as a domain.
EXTRAS_MANIFEST="$BR/.sqlite-bake-extras"
declare -a XNAMES=() XWANT=()
if [[ -n "${BAKE_EXTRA_FILES:-}" ]]; then
  IFS=',' read -r -a _extras <<< "$BAKE_EXTRA_FILES"
  for f in "${_extras[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: BAKE_EXTRA_FILES names a missing file: $f" >&2; exit 1; }
    b=$(basename "$f")
    h=$(sha256sum "$f" | cut -d' ' -f1)
    cp -f "$f" "$OVERLAY/$b"
    cp -f "$f" "$TARGET/$b"
    XNAMES+=("$b"); XWANT+=("$h")
    echo "   extra $b  $h  ($(stat -c%s "$f") bytes)"
  done
  # Prune extras this script staged before and no longer wants -- by exact name, from its own
  # manifest, so a file some other tool put in the overlay is invisible to it.
  if [[ -f "$EXTRAS_MANIFEST" ]]; then
    while read -r old; do
      [[ -n "$old" ]] || continue
      for n in "${XNAMES[@]}"; do [[ "$old" == "$n" ]] && continue 2; done
      for d in "$OVERLAY" "$TARGET"; do
        [[ -f "$d/$old" ]] && { rm -f "$d/$old"; echo "   removed extra $old ($(basename "$d"))"; }
      done
    done < "$EXTRAS_MANIFEST"
  fi
  printf '%s\n' "${XNAMES[@]}" > "$EXTRAS_MANIFEST"
fi

# Two variants that hash the same are not two variants. This has happened -- a flag that did not
# reach the compiler leaves an identical image, and the "matched pair" then measures nothing.
python3 - "${WANT[@]}" <<'PYDUP'
import sys, collections
h = sys.argv[1:]
dup = [k for k, v in collections.Counter(h).items() if v > 1]
if dup:
    sys.exit("ERROR: two or more variants are byte-identical (%s) -- a flag did not take effect"
             % ", ".join(d[:12] for d in dup))
PYDUP

> "$MANIFEST.new"
printf '%s\n' "${NAMES[@]}" >> "$MANIFEST.new"
if [[ -f "$MANIFEST" ]]; then
  echo "== pruning domains this script staged previously and no longer needs"
  IFS=',' read -r -a _keep <<< "${BAKE_KEEP:-}"
  while read -r old; do
    [[ -n "$old" ]] || continue
    for n in "${NAMES[@]}" "${_keep[@]}"; do [[ "$old" == "$n" ]] && continue 2; done
    for d in "$OVERLAY" "$TARGET"; do
      if [[ -f "$d/$old.dom" ]]; then
        rm -f "$d/$old.dom"
        echo "   removed $old.dom ($(basename "$d"))"
      fi
    done
  done < "$MANIFEST"
fi
mv -f "$MANIFEST.new" "$MANIFEST"

echo "== rebuilding the initramfs, THEN the firmware (order matters -- see header)"
make -C "$BR" build LINUX_PAYLOAD=1 A=linux-rebuild \
     CAPSTONE_CC_PATH="$(realpath "$ROOT/capstone/capstone-c")" >"$CAPSTONE_TMP_ROOT/bake-linux.log" 2>&1
make -C "$BR" build LINUX_PAYLOAD=1 A=opensbi-rebuild \
     CAPSTONE_CC_PATH="$(realpath "$ROOT/capstone/capstone-c")" >"$CAPSTONE_TMP_ROOT/bake-sbi.log" 2>&1

echo "== verifying each domain against the bytes actually inside $CPIO"
[[ -f "$CPIO" ]] || { echo "ERROR: $CPIO not found -- the initramfs rebuild produced nothing" >&2; exit 1; }
python3 - "$CPIO" "${NAMES[@]}" -- "${WANT[@]}" <<'PYVERIFY'
import sys, hashlib
cpio = sys.argv[1]
rest = sys.argv[2:]
sep = rest.index("--")
names, want = rest[:sep], rest[sep+1:]
data = open(cpio, "rb").read()

# Minimal newc cpio reader. Deliberately not `cpio -i`: extracting to a directory and hashing the
# files there reintroduces exactly the filesystem indirection this check exists to bypass.
found = {}
off = 0
while off + 110 <= len(data):
    if data[off:off+6] != b"070701":
        off += 1
        continue
    f = lambda i: int(data[off+6+i*8: off+6+i*8+8], 16)
    fsize, nsize = f(6), f(11)
    nstart = off + 110
    name = data[nstart:nstart+nsize-1].decode("ascii", "replace")
    dstart = (nstart + nsize + 3) & ~3
    dend = dstart + fsize
    if name == "TRAILER!!!":
        break
    found[name.rsplit("/", 1)[-1]] = hashlib.sha256(data[dstart:dend]).hexdigest()
    off = (dend + 3) & ~3

bad = 0
for n, w in zip(names, want):
    got = found.get(n)   # full filename: domains arrive as "<name>.dom", extras as-is
    if got is None:
        print("   %-22s NOT IN IMAGE" % n); bad += 1
    elif got != w:
        print("   %-22s STALE  image=%s want=%s" % (n, got[:16], w[:16])); bad += 1
    else:
        print("   %-22s MATCH  %s" % (n, got[:16]))
if not found:
    sys.exit("ERROR: no cpio members parsed -- the reader, not the image, is what this measured")
sys.exit(1 if bad else 0)
PYVERIFY

SIZE_AFTER=$(_img_size)
python3 - "$SIZE_BEFORE" "$SIZE_AFTER" <<'PYSIZE'
import sys
b, a = int(sys.argv[1]), int(sys.argv[2])
d = a - b
print("== initramfs %.1f MB -> %.1f MB (%+.1f MB)" % (b/1e6, a/1e6, d/1e6)
      if b else "== initramfs %.1f MB" % (a/1e6))
PYSIZE
echo "== baked: ${NAMES[*]}"
