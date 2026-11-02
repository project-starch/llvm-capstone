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
#
# Profiling (tools/profile.sh): with CAPSTONE_SPEC_PROFRUN set to the guest path of the window
# helper (profrun, see tools/profile.sh), every guest invocation runs under it, so the capprof
# plugin counts exactly that invocation in one window; the record carries the window's index
# (guest invocations so far) and the launcher writes its image record to
# $CAPSTONE_SPEC_RECORD/image-records.txt (CAPSTONE_PROFILE_RECORD) for the attribution.
# With CAPSTONE_SPEC_PLAIN=1 (config/riscv-plain.cfg) a plain RISC-V image (e_machine 0xf3) is
# accepted and runs in the guest directly, not through the launcher: route "guest-plain".
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
plain=0
if [ "$machine" = f300 ] && [ "${CAPSTONE_SPEC_PLAIN:-0}" = 1 ]; then plain=1
else [ "$machine" = 0301 ] || refuse "not a Capstone image and not a SPEC tool" "$@"; fi
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
# CAPSTONE_SPEC_STAGE=tmpfs (profiling, tools/profile.sh): the image is copied to the guest's tmpfs
# before the invocation and run from there, outside any window. On the 9p share the module cannot
# deny writers to the image (EOPNOTSUPP), so the launcher copies every image into a sealed memfd
# inside the run: 23.5 MB for gcc, most of its kernel-side cost. A local filesystem, where the
# denial works, is what a deployment has; the plain arm is staged the same way for symmetry.
staged=null
if [ "${CAPSTONE_SPEC_STAGE:-}" = tmpfs ]; then
  staged=/tmp/stage-$id-$(basename "$exe")
  env PYTHONPATH="$CAPSTONE_VM_HOST" python3 -m capstone_vm --state "$CAPSTONE_VM_STATE" exec -- sh -c "cp $(printf %q "/mnt/host/${exe#"$share"/}") $(printf %q "$staged") && chmod 0755 $(printf %q "$staged")" \
    || refuse "staging the image to the guest's tmpfs failed" "$@"
  words[0]=$staged
  staged=$(printf '"%s"' "$staged")
fi
rc=0
# A profiled invocation is one capprof window; windows are numbered in order, and only guest
# invocations open one, so this invocation's window is the count of guest invocations before it.
window=null; profile=()
redirect=
if [ -n "${CAPSTONE_SPEC_PROFRUN:-}" ]; then
  window=$(find "$inv" -maxdepth 1 -name '*.guest.json' | wc -l)   # not ls: with pipefail its "no match" exit 2 ends the script
  case "$CAPSTONE_SPEC_RECORD" in "$share"/*) ;; *) refuse "profiling: the record directory $CAPSTONE_SPEC_RECORD is not below the share" "$@" ;; esac
  # runcpu redirects this command's stdout and stderr to files in the run directory, on the host.
  # Through SSH, every line the guest prints is encrypted inside the window (the launcher's 403
  # CAPSTONE_VM_ADD_OK lines of a gcc test run cost 9.3M attempts in the SSH server alone); and
  # written straight onto the 9p share each line is a 9p round trip (5.9M kernel attempts for the
  # same 403 lines). So the guest writes to files on its tmpfs, where a line costs what a local
  # file costs a deployment, and copies them onto runcpu's files after the window closes.
  o=$(readlink -f /proc/$$/fd/1 2>/dev/null || true); e=$(readlink -f /proc/$$/fd/2 2>/dev/null || true)
  copyback=
  case "$o" in "$share"/*) redirect=">/tmp/out-$id"; copyback="cp /tmp/out-$id $(printf %q "/mnt/host/${o#"$share"/}")" ;; esac
  case "$e" in "$share"/*) redirect="$redirect 2>/tmp/err-$id"; copyback="${copyback:+$copyback; }cp /tmp/err-$id $(printf %q "/mnt/host/${e#"$share"/}")" ;; esac
  profile=(-e "CAPSTONE_PROFILE_RECORD=/mnt/host/${CAPSTONE_SPEC_RECORD#"$share"/}/image-records.txt")
  # An arm's guest environment (tools/profile.sh, arm "env"): NAME=value words, for the launcher.
  for e in ${CAPSTONE_SPEC_GUEST_ENV:-}; do profile+=(-e "$e"); done
  # The shell applies the redirections and execs the window helper, which execs the launcher: the
  # shell is gone before the window opens, and the helper's own PROFRUN_RUSAGE line lands in the
  # same stderr file. (--prefix=-c: argparse would read a bare -c as an option.)
  [ -z "$redirect" ] || profile+=(--prefix sh --prefix=-c --prefix "exec \"\$0\" \"\$@\" $redirect")
  profile+=(--prefix "$CAPSTONE_SPEC_PROFRUN")
fi
if [ $plain = 1 ]; then
  # A plain program runs as the guest's own process. The share is mounted with the host's
  # permissions (share model none), so the image executes in place and perl's t/TEST finds $^X.
  cmd="cd $(printf %q "$guest") && exec ${CAPSTONE_SPEC_GUEST_ENV:+env $CAPSTONE_SPEC_GUEST_ENV }${CAPSTONE_SPEC_PROFRUN:+$(printf %q "$CAPSTONE_SPEC_PROFRUN") }$(printf '%q ' "${words[@]}")$redirect"
  env PYTHONPATH="$CAPSTONE_VM_HOST" python3 -m capstone_vm --state "$CAPSTONE_VM_STATE" exec -- sh -c "$cmd" || rc=$?
  echo "{\"exit\": $rc}" > "$inv/$id.guest.json"
  route=guest-plain
else
  # CAPSTONE_VM_STATS makes the launcher print its counters (nodes, collections, faults, launch and elapsed
  # time) on the guest's stderr at exit, which runcpu's redirection puts in the command's .err file.
  env PYTHONPATH="$CAPSTONE_VM_HOST" python3 -m capstone_vm --state "$CAPSTONE_VM_STATE" \
    run --cwd "$guest" --result "$inv/$id.guest.json" -e CAPSTONE_VM_STATS=1 "${profile[@]}" -- "${words[@]}" || rc=$?
  route=guest
fi
if [ -n "${copyback:-}" ]; then   # the guest's output files onto runcpu's, after the window
  env PYTHONPATH="$CAPSTONE_VM_HOST" python3 -m capstone_vm --state "$CAPSTONE_VM_STATE" exec -- sh -c "$copyback; rm -f /tmp/out-$id /tmp/err-$id" \
    || refuse "copying the guest's output files back failed" "$@"
fi
vm=$(python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['identity']))" "$CAPSTONE_VM_STATE/config.json")
guest_result=$(cat "$inv/$id.guest.json" 2>/dev/null || echo null)
guest_argv=$(python3 -c "import json,sys;print(json.dumps(sys.argv[1:]))" "${words[@]}")
record $route "$rc" "{\"vm\": $vm, \"guest\": $guest_result, \"guest_argv\": $guest_argv, \"perl_pack_stub\": $stub, \"window\": $window, \"staged\": $staged}" "$@"
exit "$rc"
