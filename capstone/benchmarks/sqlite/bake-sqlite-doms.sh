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
    got = found.get(n + ".dom")
    if got is None:
        print("   %-12s NOT IN IMAGE" % n); bad += 1
    elif got != w:
        print("   %-12s STALE  image=%s want=%s" % (n, got[:16], w[:16])); bad += 1
    else:
        print("   %-12s MATCH  %s" % (n, got[:16]))
if not found:
    sys.exit("ERROR: no cpio members parsed -- the reader, not the image, is what this measured")
sys.exit(1 if bad else 0)
PYVERIFY

echo "== baked: ${NAMES[*]}"
