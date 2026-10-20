#!/bin/bash
# runcpu `submit` wrapper (config/capstone-virtual.cfg). runcpu hands every benchmark invocation,
# input generation and validation command to this script, in the run directory, with the
# command's redirections already applied on the host. It decides where the command runs:
#
#   a Capstone image (ELF e_machine 0x103)   in the guest, through capstone_vm run, same cwd
#                                            seen through the guest's share
#   SPEC's own tools (under $SPEC/bin)       on the host (specperl running specdiff)
#   525.x264_r's input generator ldecod      on the host, as the kit's host build
#   anything else                            refused (exit 125)
#
# ldecod reads past one of its own tables (tools/build-host-ldecod.sh says where); it only
# decodes the reference .264 into the .yuv that x264 encodes, and x264 itself runs on Capstone.
# For 500.perlbench_r's test workload the run directory's t/op/pack.t gets the four stubs of
# tools/perl-pack-stub.py first.
#
# Every invocation is recorded in $CAPSTONE_SPEC_RECORD/invocations/NNNN.json: route, argv,
# cwd, the executable's sha256, the guest VM's identity and the guest's exit or fault record.
set -euo pipefail
: "${SPEC:?}" "${CAPSTONE_SPEC_RECORD:?}" "${CAPSTONE_SPEC_KIT:?}"
inv=$CAPSTONE_SPEC_RECORD/invocations
mkdir -p "$inv"
# Each invocation reserves its number for good: the directory stays, so no later one reuses it.
n=0; while ! mkdir "$inv/$(printf %04d $n).reserved" 2>/dev/null; do n=$((n + 1)); done
id=$(printf %04d $n)
here=$(pwd -P)
exe=$(readlink -f "$1" 2>/dev/null || echo "$1")
record() { # route status [extra-json]
  python3 - "$inv/$id.json" "$1" "$2" "${3:-null}" "$here" "$exe" "$@" <<'PY'
import hashlib, json, sys
out, route, status, extra, cwd, exe = sys.argv[1:7]
argv = sys.argv[10:]
try: sha = hashlib.sha256(open(exe, 'rb').read()).hexdigest()
except OSError: sha = None
json.dump({"route": route, "status": int(status), "cwd": cwd, "argv": argv, "exe": exe,
           "exe_sha256": sha, "detail": json.loads(extra)}, open(out, 'w'))
PY
}
refuse() { echo "capstone-run: refused ($1): $*" >&2; record refused 125 "$(printf '"%s"' "$1")" "${@:2}"; exit 125; }

machine=$(od -An -tx1 -j18 -N2 "$exe" 2>/dev/null | tr -d ' \n' || true)
spec_bin=$(cd "$SPEC/bin" && pwd -P)
case "$(basename "$exe")" in
  ldecod_r_*)
    [ -x "${CAPSTONE_SPEC_LDECOD:-}" ] || refuse "ldecod without a host build" "$@"
    shift
    rc=0; "$CAPSTONE_SPEC_LDECOD" "$@" || rc=$?
    record host-ldecod "$rc" "$(printf '{"host_build":"%s","host_build_sha256":"%s"}' "$CAPSTONE_SPEC_LDECOD" "$(sha256sum "$CAPSTONE_SPEC_LDECOD" | cut -c1-64)")" ldecod "$@"
    exit "$rc" ;;
esac
if [ "${exe#"$spec_bin"/}" != "$exe" ] && [ "$machine" != 0301 ]; then
  rc=0; "$@" || rc=$?
  record host-spec-tool "$rc" null "$@"
  exit "$rc"
fi
[ "$machine" = 0301 ] || refuse "not a Capstone image and not a SPEC tool" "$@"
: "${CAPSTONE_VM_STATE:?}" "${CAPSTONE_VM_SHARE:?}" "${CAPSTONE_VM_HOST:?}"
share=$(cd "$CAPSTONE_VM_SHARE" && pwd -P)
case "$here" in
  "$share"/*) guest=/mnt/host/${here#"$share"/} ;;
  *) refuse "run directory $here is not below the share $share" "$@" ;;
esac
stub=null
case "$(basename "$exe")" in
  perlbench_r_*)
    if [ -f t/op/pack.t ]; then
      stub=$(python3 "$CAPSTONE_SPEC_KIT/tools/perl-pack-stub.py" t/op/pack.t) || refuse "pack.t stub" "$@"
      echo "$stub" >> "$CAPSTONE_SPEC_RECORD/perl-pack-stub.jsonl"
    fi ;;
esac
# The guest runs the same file: the image lies below the share, and the guest's cwd is the same
# directory, so a relative program path (runcpu's form, which perl's t/TEST reuses as $^X) means
# the same file there. A host path below the share, as the program or as an argument (validation
# commands name SPEC's expected outputs that way), is that file at /mnt/host in the guest.
[ "${exe#"$share"/}" != "$exe" ] || refuse "image $exe is not below the share" "$@"
words=("$@")
for i in "${!words[@]}"; do
  case "${words[$i]}" in "$share"/*) words[$i]=/mnt/host/${words[$i]#"$share"/} ;; esac
done
case "${words[0]}" in /*) [ "${words[0]#/mnt/host/}" != "${words[0]}" ] || refuse "absolute program path outside the share" "$@" ;; esac
rc=0
# CAPSTONE_VM_STATS makes the launcher print its counters (nodes, collections, faults, launch and elapsed
# time) on the guest's stderr at exit, which runcpu's redirection puts in the command's .err file.
env PYTHONPATH="$CAPSTONE_VM_HOST" python3 -m capstone_vm --state "$CAPSTONE_VM_STATE" \
  run --cwd "$guest" --result "$inv/$id.guest.json" -e CAPSTONE_VM_STATS=1 -- "${words[@]}" || rc=$?
vm=$(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['identity']))" "$CAPSTONE_VM_STATE/config.json")
guest_result=$(cat "$inv/$id.guest.json" 2>/dev/null || echo null)
guest_argv=$(python3 -c "import json,sys;print(json.dumps(sys.argv[1:]))" "${words[@]}")
record guest "$rc" "{\"vm\": $vm, \"guest\": $guest_result, \"guest_argv\": $guest_argv, \"perl_pack_stub\": $stub}" "$@"
exit "$rc"
