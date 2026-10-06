#!/bin/bash
# run-arm.sh <arm> [out]: boot once with that arm's image, run all 21 cases.
#
# The arm is the heap the image links. This script REFUSES to start unless the
# image it is about to boot is the one build-arms.sh recorded for this arm:
# three earlier rounds in this lane ran bounds-only under a sublet label,
# because the label came from an argument and nothing checked the binary.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../../.." && pwd)}
KIT=${KIT:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/cpython-arms}
CORPUS=$HERE/..
arm=${1:?usage: run-arm.sh <arm> [out]}
OUT=${2:-$KIT/results/$arm-$(date -u +%Y%m%d-%H%M%S)}
RT=$REPO/capstone/runtime
export PYTHONPATH=$RT/host CAPSTONE_QEMU_LOCK=$KIT/qemu.lock
# The revoking arms spend far more revocation nodes than the 65536 default. An
# earlier round at the default produced 44 cause-30 (INSUF_RESOURCES) rows out
# of 54, which is a resource ceiling and not a verdict about any defect.
# Which arms revoke, and therefore need more than the 65536-node default. The
# arm NAMES are read from the corpus so this cannot drift again: the first
# version of this line matched sysalloc-sublet and sublet-pymalloc, names the
# corpus stopped using, so the `sublet` arm would silently have run at the
# default and produced cause-30 (INSUF_RESOURCES) rows -- a resource ceiling,
# not a verdict about any defect.
case $arm in
  sublet|sysalloc-sublet|sublet-pymalloc|sublet-gc)
    export CAPSTONE_REV_NODES=${CAPSTONE_REV_NODES:-16777216} ;;
esac
# The arm must be one this corpus declares, so a typo cannot produce a result
# directory that analyse.py will later refuse.
DECL=$(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['required_arms']))" \
       "$CORPUS/corpus.json")
case " $DECL " in *" $arm "*) ;; *)
  echo "arm '$arm' is not in corpus.json required_arms: $DECL" >&2; exit 2 ;;
esac

IMG=$KIT/images/python-$arm.dom
INPUTS=$KIT/images/inputs.tsv
[ -f "$IMG" ]    || { echo "no image for $arm -- run build-arms.sh $arm" >&2; exit 2; }
[ -f "$INPUTS" ] || { echo "no $INPUTS -- run build-arms.sh" >&2; exit 2; }
want=$(awk -v a="$arm" '$1==a{print $3}' "$INPUTS")
have=$(sha256sum "$IMG" | cut -d' ' -f1)
[ -n "$want" ] || { echo "$arm is not in $INPUTS" >&2; exit 2; }
[ "$want" = "$have" ] || {
  echo "REFUSING: $IMG is not the image recorded for $arm" >&2
  echo "  recorded ${want:0:16}  present ${have:0:16}" >&2; exit 2; }

mkdir -p "$OUT"
cp "$INPUTS" "$OUT/inputs.tsv"
printf 'arm\t%s\nimage_sha256\t%s\nrev_nodes\t%s\nstarted_utc\t%s\n' \
  "$arm" "$have" "${CAPSTONE_REV_NODES:-65536}" "$(date -u +%FT%TZ)" > "$OUT/run.meta"

# ---- stage the cases ----------------------------------------------------
SHARE=$KIT/share-$arm; rm -rf "$SHARE"; mkdir -p "$SHARE/cases"
: > "$SHARE/cases.list"
for d in "$CORPUS"/[0-9][0-9]_*/; do
  n=$(basename "$d")
  mkdir -p "$SHARE/cases/$n"
  cp "$d/trigger.py" "$SHARE/cases/$n/"
  [ -f "$d/upstream_test.py" ] && cp "$d/upstream_test.py" "$SHARE/cases/$n/"
  echo "$n" >> "$SHARE/cases.list"
done
cp "$IMG" "$SHARE/python.dom"
cp "$HERE/run-all.sh" "$SHARE/"
echo "staged $(wc -l < "$SHARE/cases.list") cases into $SHARE"

# ---- boot, run, bring down even on failure ------------------------------
ST=$KIT/vm-$arm
vm() { python3 -m capstone_vm --state "$ST" "$@"; }
cleanup() { vm down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM HUP     # armed BEFORE the VM exists, not after
vm down >/dev/null 2>&1 || true
vm up --share "$SHARE" >"$OUT/boot.log" 2>&1 || { echo "boot failed, see $OUT/boot.log" >&2; exit 2; }
vm run --cwd /mnt/host -- sh /mnt/host/run-all.sh /mnt/host/python.dom "${CASE_BUDGET:-120}" \
    < "$SHARE/cases.list" > "$OUT/stream.txt" 2>&1
rc=$?
printf 'ended_utc\t%s\nstream_rc\t%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$OUT/run.meta"

# ---- a stream with no ARMS-DONE is not a result -------------------------
grep -q '^ARMS-BEGIN' "$OUT/stream.txt" || { echo "no ARMS-BEGIN: the guest never started the batch" >&2; exit 2; }
grep -q '^ARMS-DONE'  "$OUT/stream.txt" || echo "WARNING: no ARMS-DONE -- the stream was cut, rows after the last CASE are missing" >&2
awk -v arm="$arm" 'BEGIN{OFS="\t"; print "case","arm","rc","last"}
  /^CASE /{ n=$2; rc=""; sub(/^rc=/,"",$3); rc=$3; $1=$2=$3=""; sub(/^ *LAST= */,""); print n,arm,rc,$0 }' \
  "$OUT/stream.txt" > "$OUT/verdicts.tsv"
echo "rows: $(($(wc -l < "$OUT/verdicts.tsv")-1)) / $(wc -l < "$SHARE/cases.list")"
echo "out:  $OUT"
