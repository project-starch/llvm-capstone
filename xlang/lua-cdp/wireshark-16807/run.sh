#!/usr/bin/env bash
# CHECK Wireshark #16807 by DIFFERENTIAL, on the stock apt tshark under valgrind.
#
# The bug: a Lua TvbRange (buffer(0,16)) cached in a global table outlives the C
# `tvbuff` it wraps. On re-dissection of the same packet the C engine has freed
# that tvbuff (epan_dissect_reset -> tvb_free_chain); the stale TvbRange is then
# handed to proto_tree_add_item_new -> tvb_ensure_bytes_exist -> UAF read.
#
# Headless re-dissection is produced with two-pass analysis (-2): pass 1 stashes
# the TvbRange and frees its tvbuff; pass 2 re-dissects the same packet number
# and reuses the stale range. tshark has no GUI but -2 exercises the identical
# re-dissection path the issue hits by switching packets in the GUI.
#
# PASS iff all three hold:
#   A. VULN trigger + two-pass  -> valgrind Invalid read with the tvb-UAF stack
#      signature (tvb_ensure_bytes_exist, freed by tvb_free_chain/epan_dissect_reset).
#   B. VULN trigger + single-pass (no re-dissection) -> signature ABSENT (control).
#   C. FIXED trigger + two-pass (no cached range)    -> signature ABSENT (control).
# Any one failing = FAIL.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

command -v tshark   >/dev/null 2>&1 || { echo "run ./build.sh first (no tshark)"   >&2; exit 2; }
command -v valgrind >/dev/null 2>&1 || { echo "run ./build.sh first (no valgrind)" >&2; exit 2; }

# The apt tshark AppArmor profile (/etc/apparmor.d/tshark) confines /usr/bin/tshark
# to reading /usr/share/wireshark/** and /tmp (abstractions/user-tmp) only -- it
# has no @{HOME} rule, so a pcap/lua under the project path is denied. Stage the
# runtime files in a /tmp workdir, which the profile permits.
RT=$(mktemp -d /tmp/ws16807.XXXXXX)
trap 'rm -rf "$RT"' EXIT
cp "$HERE/trigger.lua" "$HERE/trigger_fixed.lua" "$RT/"
python3 "$HERE/gen_pcap.py" "$RT/ftp.pcap"

vg() { # $1=lua $2..=extra tshark args -> stderr of the run
  local lua="$1"; shift
  valgrind -q --error-exitcode=99 --leak-check=no --errors-for-leak-kinds=none \
    tshark -r "$RT/ftp.pcap" -X lua_script:"$RT/$lua" "$@" 2>&1 >/dev/null
}

# tvb-UAF stack signature: an Invalid read in tvb_ensure_bytes_exist whose block
# was freed by tvb_free_chain via epan_dissect_reset. Noise-independent (does not
# rely on the valgrind exit code, which unrelated plugin-init warnings pollute).
sig() { grep -q 'Invalid read' <<<"$1" && grep -q 'tvb_ensure_bytes_exist' <<<"$1" \
        && grep -q 'tvb_free_chain' <<<"$1" && grep -q 'epan_dissect_reset' <<<"$1"; }

echo "== A: VULN trigger, two-pass (-2) =="
A=$(vg trigger.lua -2);        sig "$A" && a=1 || a=0; echo "   tvb-UAF signature present: $a"
echo "== B: VULN trigger, single-pass (control) =="
B=$(vg trigger.lua);          sig "$B" && b=1 || b=0; echo "   tvb-UAF signature present: $b"
echo "== C: FIXED trigger, two-pass (control) =="
C=$(vg trigger_fixed.lua -2); sig "$C" && c=1 || c=0; echo "   tvb-UAF signature present: $c"

echo "--- verdict: A(vuln,2pass)=$a  B(vuln,1pass)=$b  C(fixed,2pass)=$c ---"
if [ "$a" = 1 ] && [ "$b" = 0 ] && [ "$c" = 0 ]; then
  echo "PASS: Wireshark #16807 reproduced (tvb UAF only when a cached Lua TvbRange"
  echo "      is reused across re-dissection; clean without re-dissection and with"
  echo "      the fixed no-cache script)."
  # show the captured UAF block for the record
  echo "$A" | grep -aA22 'Invalid read of size 1' | grep -aE 'Invalid read|tvb_|proto_tree_add_item|epan_dissect|freed|alloc' | head -12
  exit 0
fi
echo "FAIL: differential not satisfied" >&2
exit 1
