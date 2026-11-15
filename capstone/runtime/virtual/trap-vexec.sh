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
# kernel's code= word, else the loader's dom_last_base. The record is the newest one that names
# the child's pid (a fault of another thread of the child names that thread's id and prints no line). As before, the line goes to stderr when stderr is a terminal or
# CAPSTONE_EXEC_DIAGNOSTICS is set, and to the file CAPSTONE_FAULT_RECORD names when that is set.
# A signal death with no Sublet record (Linux's own trap path: an illegal instruction, a page
# fault) gets a diagnostic line `capstone-exec: unhandled signal=...` on stderr only; see unhandled().
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

    # One read of the kernel log serves both reports: each process start costs a second or more
    # under QEMU, so the reports parse in the shell and start nothing else.
    log=$(dmesg | grep -E "$RECORDS|sublet: kernel fault |sublet: .*\[$child\] enters |\[$child\]: unhandled signal |\] epc : |\] status: ")
    kernel_faults "$child" "$image"
    if [ $rc -le 128 ] || [ $rc -gt 192 ]; then
        exit $rc
    fi
    report "$child" "$image"
    signal=$((rc - 128))
    trap - $signal 2>/dev/null
    kill -$signal $$
    exit $rc
}

# emit LINE: the launcher's line, to stderr and the fault record as the launcher of the deleted route did.
emit() {
    if [ -n "$CAPSTONE_FAULT_RECORD" ]; then
        echo "$1" >> "$CAPSTONE_FAULT_RECORD"
    fi
    emit_stderr "$1"
}

# kernel_faults PID IMAGE: one line per kernel access through the child's reference that faulted,
#   sublet: kernel fault cause=C address=A nr=N pid=PID comm=NAME
# printed as
#   capstone-exec: kernel fault cause=C address=A nr=N image=IMAGE
# The kernel refused the access (EFAULT or a short count) and the program went on, so these lines
# stand beside the exit status: the judges count such a run as blocked in the kernel, apart from
# caught and missed.
kernel_faults() {
    echo "$log" | while IFS= read -r line; do
        case $line in
            *"sublet: kernel fault "*" pid=$1 "*) ;;
            *) continue ;;
        esac
        cause= address= nr=
        for word in ${line#*sublet: kernel fault }; do
            case $word in
                cause=*) cause=${word#cause=} ;;
                address=*) address=${word#address=} ;;
                nr=*) nr=${word#nr=} ;;
            esac
        done
        emit "capstone-exec: kernel fault cause=$cause address=$address nr=$nr image=${2##*/}"
    done
}

# report PID IMAGE: the launcher's fault line from the newest kernel fault record for PID. Words the
# record lacks read 0x0; the entry falls back to the loader's `enters` line for PID, the code base to
# the loader's dom_last_base.
report() {
    record= entered=
    while IFS= read -r line; do
        case $line in
            *"sublet: fault "*" pid=$1 "*|*"capstone_task: fault "*" pid=$1 "*) record=$line ;;
            *"[$1] enters 0x"*) entered=${line##* enters }; entered=${entered%%,*} ;;
        esac
    done <<EOF_LOG
$log
EOF_LOG
    if [ -z "$record" ]; then
        unhandled "$1" "$2"
        return 0
    fi

    base=0
    read -r base < "$PARAMETERS/dom_last_base" 2>/dev/null
    case $base in
        ''|*[!0-9]*) base=0 ;;
    esac
    base=$(printf '0x%x' "$base")

    cause= pc= address= ra=0x0 fp=0x0 entry=${entered:-0x0} code=$base-$base
    for word in ${record#*fault }; do
        case $word in
            cause=*) cause=${word#cause=} ;;
            pc=*) pc=${word#pc=} ;;
            address=*) address=${word#address=} ;;
            ra=*) ra=${word#ra=} ;;
            fp=*) fp=${word#fp=} ;;
            entry=*) entry=${word#entry=} ;;
            code=*) code=${word#code=} ;;
        esac
    done
    emit "capstone-exec: domain fault cause=$cause pc=$pc address=$address ra=$ra fp=$fp entry=$entry code=$code image=$2"
}

# unhandled PID IMAGE: a signal death that the kernel logged with no Sublet record. A trap outside
# the Sublet causes 24 to 30 (an illegal instruction, a page fault) takes Linux's own path, which
# logs, when the program has no handler for the signal,
#   COMM[PID]: unhandled signal S code 0xC at 0xADDR in ...
#   epc : PC ra : RA sp : ...
#   status: ... badaddr: ... cause: CAUSE
# and the script prints it as
#   capstone-exec: unhandled signal=S code=0xC cause=N pc=0x.. address=0x.. ra=0x.. image=IMAGE
# The line is a diagnostic: it goes to stderr only (not to CAPSTONE_FAULT_RECORD), and the
# judges, which read `domain fault` lines, do not count it.
unhandled() {
    found= signal= code= address= pc= ra= cause=
    while IFS= read -r line; do
        case $line in
            *"[$1]: unhandled signal "*)
                found=1 pc= ra= cause=
                signal_words ${line#*"[$1]: unhandled signal "} ;;
            *"] epc : "*)
                [ -n "$found" ] && [ -z "$pc" ] || continue
                epc_words ${line#*"] epc : "} ;;
            *"] status: "*" cause: "*)
                [ -n "$found" ] && [ -z "$cause" ] || continue
                cause=${line##* cause: } ;;
        esac
    done <<EOF_LOG
$log
EOF_LOG
    [ -n "$found" ] || return 0
    emit_stderr "capstone-exec: unhandled signal=$signal code=$code cause=$(hex_dec "$cause") pc=$(hex "$pc") address=$(hex "$address") ra=$(hex "$ra") image=$2"
}

# signal_words S code 0xC at 0xADDR ...; epc_words PC ra : RA ...
signal_words() { signal=$1 code=$3 address=$5; }
epc_words() { pc=$1 ra=$4; }

emit_stderr() {
    if [ -t 2 ] || [ -n "$CAPSTONE_EXEC_DIAGNOSTICS" ]; then
        echo "$1" >&2
    fi
}

# hex WORD: Linux's zero-padded register word (with or without 0x) as 0x.. without the padding; empty stays ?.
hex() {
    case ${1#0x} in
        '') echo '?' ;;
        *[!0-9a-fA-F]*) echo '?' ;;
        *) printf '0x%x' "0x${1#0x}" ;;
    esac
}
hex_dec() {
    case ${1#0x} in
        ''|*[!0-9a-fA-F]*) echo '?' ;;
        *) printf '%d' "0x${1#0x}" ;;
    esac
}

main "$@"
