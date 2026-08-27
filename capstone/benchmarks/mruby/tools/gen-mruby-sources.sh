#!/usr/bin/env bash
# Run mruby's HOST rake to produce the three things the domain cannot get any
# other way, then stop. We do not cross-compile through rake.
#
#     tools/gen-mruby-sources.sh          # fetch/patch if needed, then generate
#
#   * the presymbol tables -- 22 of 32 core files do not compile without them,
#     and each failure is a missing MRB_SYM__* rather than anything about the
#     target;
#   * mrblib.c and gem_init.c, the standard library as bytecode;
#   * build/host/amalgam/mruby.c, ONE translation unit, which the gp-captable ABI
#     requires rather than prefers: getGpCaptableIndex numbers globals per module.
#
# build-mruby-silicon.sh names this script when the amalgamation is missing, and
# for a while it did not exist -- so a failed build pointed at nothing. It also
# COPIES THE PATCHES IN: the amalgamation is generated once and then reused, so a
# source edit that skips this step is silently absent from every later build.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MRUBY_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$MRUBY_DIR/../../tests/capstone-test-env.sh"

SRC=${MRUBY_SRC_DIR:-$CAPSTONE_TMP_ROOT/mruby-src}
[[ -d "$SRC/.git" ]] || bash "$MRUBY_DIR/fetch-mruby.sh"

command -v rake >/dev/null || { echo "rake not on PATH -- the HOST build needs ruby+rake" >&2; exit 2; }

# Regenerate from scratch. An incremental rake happily keeps a stale amalgam.
rm -rf "$SRC/build/host/amalgam"
env -C "$SRC" MRUBY_CONFIG="$MRUBY_DIR/mruby_build_config_capstone.rb" rake amalgam

AMALGAM="$SRC/build/host/amalgam/mruby.c"
[[ -s "$AMALGAM" ]] || { echo "rake finished but $AMALGAM is missing or empty" >&2; exit 1; }
echo "amalgamation: $AMALGAM ($(grep -c '' "$AMALGAM") lines)"

# The patches are the point of regenerating, so prove they are IN the output
# rather than assuming the tree was patched. A "no data" here is an error, not a
# zero: a build whose capability patch silently vanished looks fine and is not.
python3 - "$SRC/build/host/amalgam" <<'PY'
import sys, pathlib
d = pathlib.Path(sys.argv[1])
src = (d / "mruby.c").read_text(encoding="utf8", errors="replace")
hdr = (d / "mruby.h").read_text(encoding="utf8", errors="replace")

# BOTH files, because rake splits the patched sources across them: the
# embedded-string width is a header constant while the alignment fixes are in the
# .c, and a check that looks in one file reports the other as missing. That is not
# hypothetical -- it is what this check did on its first run.
checks = [
    ("0001 embedded-string width", hdr.count("#define MRB_STR_EMBED_LEN_BITS 6"), 1),
    ("0001 capability alignment",  src.count("mrb_alignas(sizeof(void*))"),        4),
    ("0003 stack-bounds probe",    src.count("md_probe_stack"),                    2),
]
bad = []
for name, got, want in checks:
    ok = got == want
    print(f"  {'ok  ' if ok else 'WRONG'} {name}: {got} site(s), expected {want}")
    if not ok:
        bad.append(name)
if bad:
    sys.exit("patch(es) not correctly present: " + ", ".join(bad))
PY
