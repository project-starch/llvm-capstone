#!/usr/bin/env bash
# Fetch and pin the FULL SQLite source tree, for the one file the amalgamation does not carry:
# test/speedtest1.c.
#
# WHY A SECOND FETCH. fetch-sqlite.sh gets the amalgamation zip, which is the engine only. The
# canonical benchmark ships in the full distribution. Both must be the SAME VERSION or the benchmark
# and the engine under test disagree about what they are measuring, so SQLITE_VERSION is shared and
# the check below is not optional.
#
# The tree was on this machine under /tmp before this script existed, which is one reboot from
# unreproducible -- the same reason the amalgamation is pinned rather than assumed.
#
# PINNED ON THE CHECKIN UUID, NOT ON THE TARBALL'S BYTES (changed 2026-09-29). This URL is the
# FOSSIL tarball endpoint, and Fossil REGENERATES that archive per request: its bytes are a property
# of one compression run, not of the source. The previous SHA3-256 pin therefore failed on every
# cold cache -- fetch a tree that is provably correct and the gate rejects it, which reads exactly
# like tampering. Measured 2026-09-29: the pinned 03a50b97... against an actual f19a1884..., on a
# 12,786,316-byte archive whose contents were the right checkin.
#
# manifest.uuid IS a property of the source: it names the Fossil checkin the tree was exported from,
# and it is identical across regenerations. It is also the SAME identity SQLite itself reports at
# run time -- speedtest1 prints "3.53.3 2026-06-26 20:14:12 d4c0e51e4aeb..." -- so a recorded
# benchmark result can be tied back to this pin without trusting the download at all.
#
# The tarball hash is still COMPUTED and PRINTED, because it is useful evidence in a log. It is no
# longer a failure condition, because it cannot be one.
#
# NOTE ON THE OTHER FETCHER: fetch-sqlite.sh keeps its SHA3 pin and SHOULD. It downloads a RELEASED
# amalgamation zip from sqlite.org/<year>/, which is a published artifact and byte-stable -- its
# committed pin still verified on 2026-09-29 -- and that zip contains no manifest.uuid to pin on.
# Same-looking gate, different kind of artifact.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

# Same version as fetch-sqlite.sh, expressed the way the full distribution names itself:
# the amalgamation is 3530300, the source tarball is 3.53.3.
SQLITE_VERSION=${SQLITE_VERSION:-3530300}
SQLITE_SRC_VERSION=${SQLITE_SRC_VERSION:-3.53.3}
SQLITE_SRC_UUID=${SQLITE_SRC_UUID:-d4c0e51e4aeb96955b99185ab9cde75c339e2c29c3f3f12428d364a10d782c62}
SQLITE_SRC_YEAR=${SQLITE_SRC_YEAR:-2026}
SQLITE_FULL_ROOT=${SQLITE_FULL_ROOT:-$CAPSTONE_TMP_ROOT/sqlite-src-full}
SQLITE_FULL_ARCHIVE=${SQLITE_FULL_ARCHIVE:-$SQLITE_FULL_ROOT/sqlite-$SQLITE_SRC_VERSION.tar.gz}
SQLITE_FULL_URL=${SQLITE_FULL_URL:-https://sqlite.org/src/tarball/sqlite.tar.gz?r=version-$SQLITE_SRC_VERSION}

mkdir -p "$SQLITE_FULL_ROOT"

if [[ ! -f "$SQLITE_FULL_ARCHIVE" ]]; then
  curl -fL "$SQLITE_FULL_URL" -o "$SQLITE_FULL_ARCHIVE"
fi

# THE EXTRACTED DIRECTORY NAME IS NOT ASSUMED. It used to be hardcoded as sqlite-version-<v>; the
# Fossil endpoint now produces sqlite/. Both were observed from the same URL, so the name is as much
# a property of the endpoint's mood as the bytes are: take it from the archive instead. An explicit
# SQLITE_FULL_DIR still wins, for a caller that stages the tree itself.
if [[ -n "${SQLITE_FULL_DIR:-}" ]]; then
  _dir=$SQLITE_FULL_DIR
else
  # NOT `tar tzf ... | head -1`: head exits after one line, tar takes SIGPIPE, and under
  # `set -o pipefail` the script dies with 141 -- which reads as a fetch failure and is not one.
  # List to a file instead, so nothing is killed mid-write.
  _listing=$(mktemp)
  tar tzf "$SQLITE_FULL_ARCHIVE" > "$_listing"
  _top=$(sed -n 1p "$_listing" | cut -d/ -f1)
  rm -f "$_listing"
  [[ -n "$_top" ]] || { echo "ERROR: cannot read the archive's top-level directory" >&2; exit 1; }
  _dir=$SQLITE_FULL_ROOT/$_top
fi

if [[ ! -f "$_dir/test/speedtest1.c" ]]; then
  tar xzf "$SQLITE_FULL_ARCHIVE" -C "$SQLITE_FULL_ROOT"
fi

# THE PIN. Fails closed: no manifest.uuid, or the wrong one, and nothing downstream runs.
_uuid_file=$_dir/manifest.uuid
[[ -f "$_uuid_file" ]] || {
  echo "ERROR: no manifest.uuid in $_dir -- cannot identify the checkin, refusing to proceed" >&2
  exit 1; }
_have_uuid=$(tr -d ' \n' < "$_uuid_file")
[[ "$_have_uuid" == "$SQLITE_SRC_UUID" ]] || {
  echo "ERROR: SQLite checkin UUID mismatch: expected $SQLITE_SRC_UUID, got $_have_uuid" >&2
  exit 1; }

# Informational only: what this particular tarball hashed to. Never a gate -- see the header.
echo "fetch-sqlite-src: checkin $_have_uuid (tarball sha3-256 $(python3 -c '
import hashlib,sys; print(hashlib.sha3_256(open(sys.argv[1],"rb").read()).hexdigest())' "$SQLITE_FULL_ARCHIVE"))" >&2

# THE VERSION AGREEMENT CHECK, and it is the point of the file. The benchmark and the engine must
# come from the same release; a silent drift here would make every comparison a comparison of two
# SQLites. The full tree states its version in VERSION, the amalgamation in its own header.
if [[ -f "$_dir/VERSION" ]]; then
  _have=$(tr -d ' \n' < "$_dir/VERSION")
  [[ "$_have" == "$SQLITE_SRC_VERSION" ]] || {
    echo "ERROR: full source is $_have, expected $SQLITE_SRC_VERSION" >&2; exit 1; }
fi

for required in test/speedtest1.c VERSION; do
  [[ -e "$_dir/$required" ]] || {
    echo "ERROR: $_dir/$required missing after extract" >&2; exit 1; }
done

echo "$_dir"
