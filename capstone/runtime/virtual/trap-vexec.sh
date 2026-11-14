#!/bin/sh
# capstone-vexec on the trap route (trap_platform.py puts it into every kit as adapter/capstone-vexec).
#
#   capstone-vexec IMAGE [ARGUMENT...]
#   capstone-vexec --platform
#   capstone-vexec --stats
#
# The kernel runs a .dom image itself (binfmt, sublet.dom_binfmt=1), so this script holds no
# device and no context. The runners start every image as `./capstone-vexec IMAGE ...`, as on
# the deleted route. The script runs the image as its child and ends the way the child ended:
# the same exit status, or death by the same signal. When the child dies by a signal and the
# kernel logged a Sublet fault for it,
#
#   sublet: fault cause=C pc=0x.. address=0x.. pid=N comm=NAME
#
# the script prints the launcher's line that the corpus judges parse,
#
#   capstone-exec: domain fault cause=C pc=0x.. address=0x.. ra=0x.. fp=0x.. entry=0x.. code=0x..-0x.. image=IMAGE
#
# Words the kernel line lacks read 0x0. The entry is the kernel's entry= word, else the address
# of the loader's line `sublet: COMM[PID] enters 0x..` for the child. The code base is the
# kernel's code= word, else the loader's dom_last_base. As before, the line goes to stderr when stderr is a terminal or
# CAPSTONE_EXEC_DIAGNOSTICS is set, and to the file CAPSTONE_FAULT_RECORD names when that is set.
#
# The child runs as an asynchronous command, so it starts with SIGINT and SIGQUIT ignored.
# HUP, INT, TERM, USR1, USR2 and ALRM sent to this script go on to the child.
#
# --platform prints the loader's parameters and exits 0 only when the kernel runs protected programs.
# --stats prints one JSON object: every numeric parameter of the kernel's sublet and captag code
# (table growths, node pages, tag copies and so on) as "module.name": value, with "version": 2.
# The deleted route's object (live domains, nodes) has no counterpart; readers record it only.

PARAMETERS=${SUBLET_PARAMETERS:-/sys/module/sublet/parameters}   # the variable is a test hook
RECORDS='sublet: fault |capstone_task: fault '

main() {
    if [ "$1" = --platform ]; then
        platform
        exit $?
    fi
    if [ "$1" = --stats ]; then
        stats
        exit $?
    fi
    if [ $# -lt 1 ]; then
        echo "usage: capstone-vexec IMAGE [ARGUMENT...] | capstone-vexec --platform" >&2
        exit 2
    fi
    run "$@"
}

platform() {
    for name in dom_binfmt dom_wide_grants dom_local_tmp dom_null dom_last_base; do
        value=$(cat "$PARAMETERS/$name" 2>/dev/null)
        echo "CAPSTONE_PLATFORM $name=${value:-absent}"
    done
    if [ "$(cat "$PARAMETERS/dom_binfmt" 2>/dev/null)" = Y ] && dmesg | grep -q 'sublet: .dom loader registered'; then
        return 0
    fi
    echo "capstone-vexec: this kernel does not run protected programs (sublet.dom_binfmt=1)" >&2
    return 1
}

stats() {
    printf '{"version":2,"route":"trap"'
    for file in ${SUBLET_PARAMETERS:-/sys/module/sublet/parameters}/* /sys/module/captag/parameters/*; do
        [ -r "$file" ] || continue
        value=$(cat "$file" 2>/dev/null)
        case $value in
            ''|*[!0-9]*) continue ;;
        esac
        module=${file%/parameters/*}
        printf ',"%s.%s":%s' "${module##*/}" "${file##*/}" "$value"
    done
    printf '}\n'
}

run() {
    image=$1
    shift
    case $image in
        */*) ;;
        *) image=./$image ;;
    esac

    seen=$(dmesg | grep -cE "$RECORDS")
    exec 3<&0
    "$image" "$@" <&3 3<&- &
    child=$!
    exec 3<&-
    for s in HUP INT TERM USR1 USR2 ALRM; do
        trap "kill -$s $child 2>/dev/null" $s
    done

    while :; do
        wait $child
        rc=$?
        kill -0 $child 2>/dev/null || break
    done
    trap - HUP INT TERM USR1 USR2 ALRM

    if [ $rc -le 128 ] || [ $rc -gt 192 ]; then
        exit $rc
    fi
    report "$child" "$image" "$seen"
    signal=$((rc - 128))
    trap - $signal 2>/dev/null
    kill -$signal $$
    exit $rc
}

# report PID IMAGE SEEN: print the launcher's fault line from the first new kernel record for PID,
# or from the first new record when none names PID (a thread's record names its own id).
report() {
    records=$(dmesg | grep -E "$RECORDS" | tail -n +$(($3 + 1)))
    [ -n "$records" ] || return 0
    record=$(echo "$records" | grep " pid=$1 " | head -n 1)
    [ -n "$record" ] || record=$(echo "$records" | head -n 1)

    base=$(cat "$PARAMETERS/dom_last_base" 2>/dev/null)
    case $base in
        ''|*[!0-9]*) base=0 ;;
    esac
    base=$(printf '0x%x' "$base")
    entered=$(dmesg | grep "sublet: .*\[$1\] enters 0x" | tail -n 1 | sed 's/.* enters \(0x[0-9a-f]*\).*/\1/')

    line=$(echo "$record" | awk -v image="$2" -v base="$base" -v entered="${entered:-0x0}" '{
        for (i = 1; i <= NF; i++) {
            k = $i; v = $i
            if (sub(/=.*/, "", k) && sub(/^[^=]*=/, "", v)) f[k] = v
        }
        ra = ("ra" in f) ? f["ra"] : "0x0"
        fp = ("fp" in f) ? f["fp"] : "0x0"
        entry = ("entry" in f) ? f["entry"] : entered
        code = ("code" in f) ? f["code"] : base "-" base
        printf "capstone-exec: domain fault cause=%s pc=%s address=%s ra=%s fp=%s entry=%s code=%s image=%s\n",
               f["cause"], f["pc"], f["address"], ra, fp, entry, code, image
    }')

    if [ -n "$CAPSTONE_FAULT_RECORD" ]; then
        echo "$line" >> "$CAPSTONE_FAULT_RECORD"
    fi
    if [ -t 2 ] || [ -n "$CAPSTONE_EXEC_DIAGNOSTICS" ]; then
        echo "$line" >&2
    fi
}

main "$@"
