#!/bin/sh
# test-vexec.sh: trap-vexec.sh on the host, with a fake dmesg and parameter directory.
#   1. An image that the "kernel" kills with a Sublet fault: the launcher line, death by SIGSEGV (139).
#   2. An image that exits 3: exits 3 and prints the launcher's identity line, `capstone-exec: exit image=IMAGE
#      sha256=HEX`, the image's own digest. Control: without CAPSTONE_EXEC_DIAGNOSTICS (stderr not a terminal) no line.
#      Control: the faulting image of 1 prints no exit line.
#   3. Control: --platform refuses when dom_binfmt is not Y.
#   4. --stats prints the numeric parameters as JSON and skips the others.
#   5. A kernel access fault of the child: the kernel-fault line, and the child's own exit status 0.
#      Control: another pid's kernel fault prints nothing.
#   6. A death by SIGILL that Linux logged as an unhandled signal, with no Sublet record: the
#      diagnostic line on stderr, nothing in CAPSTONE_FAULT_RECORD, no domain fault line, exit 132.
#      Control: another pid's unhandled signal is ignored.
set -u
here=$(cd "$(dirname "$0")" && pwd)
vexec=$here/../../trap-vexec.sh
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin" "$work/sublet/parameters"
echo 274877906944 > "$work/sublet/parameters/dom_last_base"   # 0x4000000000
echo N > "$work/sublet/parameters/dom_binfmt"
printf '#!/bin/sh\ncat "%s/log" 2>/dev/null\n' "$work" > "$work/bin/dmesg"
printf '[ 0.5] sublet: fault cause=28 pc=0x1 address=0x2 pid=1 comm=old\n' > "$work/log"
# a shell may print its own "Segmentation fault" note: the test reads only the launcher line
cat > "$work/fault.dom" <<IMG
#!/bin/sh
echo "[ 0.9] sublet: fault.dom[\$\$] enters 0x4000000100, table 0x8000" >> "$work/log"
echo "[ 1.0] sublet: fault cause=25 pc=0x4000001010 address=0x55 pid=\$\$ comm=fault.dom" >> "$work/log"
kill -SEGV \$\$
IMG
printf '#!/bin/sh\nexit 3\n' > "$work/three.dom"
cat > "$work/kfault.dom" <<IMG
#!/bin/sh
echo "[ 2.0] sublet: kernel fault cause=25 address=0x4000002000 nr=63 pid=\$\$ comm=kfault.dom" >> "$work/log"
echo "[ 2.1] sublet: kernel fault cause=28 address=0x1 nr=64 pid=999999 comm=other" >> "$work/log"
exit 0
IMG
chmod +x "$work/kfault.dom"
cat > "$work/ill.dom" <<IMG
#!/bin/sh
cat >> "$work/log" <<LOG
[ 3.0] other[999999]: unhandled signal 11 code 0x1 at 0x0000000000000008 in other[10000+1000]
[ 3.0] epc : 0000000000010010 ra : 0000000000010020 sp : 0000003ff0000000
[ 3.0] status: 0000000200004020 badaddr: 0000000000000008 cause: 000000000000000d
[ 3.1] ill.dom[\$\$]: unhandled signal 4 code 0x1 at 0x0000003ff7f4fadc in ill.dom[3ff7e00000+174000]
[ 3.1] CPU: 0 PID: \$\$ Comm: ill.dom Not tainted 6.1.0 #46
[ 3.1] epc : 0000003ff7f4fadc ra : 0000003ff7f3fac4 sp : 0000003ff7dff9e0
[ 3.1]  gp : 0000003ff7e00000 tp : 0000003ff7cff000 t0 : 0000000000000001
[ 3.1] status: 0000000200004020 badaddr: 00000000c0001073 cause: 0000000000000002
LOG
kill -ILL \$\$
IMG
chmod +x "$work/ill.dom"
chmod +x "$work/bin/dmesg" "$work/fault.dom" "$work/three.dom"
fail=0
export PATH="$work/bin:$PATH" SUBLET_PARAMETERS="$work/sublet/parameters" CAPSTONE_EXEC_DIAGNOSTICS=1

out=$(cd "$work" && sh "$vexec" fault.dom 2>&1 | grep "domain fault")
rc=$(cd "$work" && sh "$vexec" fault.dom > /dev/null 2>&1; echo $?)
want="capstone-exec: domain fault cause=25 pc=0x4000001010 address=0x55 ra=0x0 fp=0x0 entry=0x4000000100 code=0x4000000000-0x4000000000 image=./fault.dom"
if [ "$rc" = 139 ] && [ "$out" = "$want" ]; then echo "PASS fault line and SIGSEGV"; else echo "FAIL fault: rc=$rc out=$out"; fail=1; fi

out=$(cd "$work" && sh "$vexec" fault.dom 2>&1 | grep "exit image")
if [ -z "$out" ]; then echo "PASS control: a fault prints no exit line"; else echo "FAIL fault exit line: $out"; fail=1; fi

three=$(sha256sum "$work/three.dom" | cut -d' ' -f1)
out=$(cd "$work" && sh "$vexec" three.dom 2>&1)
rc=$?
want="capstone-exec: exit image=./three.dom sha256=$three"
if [ "$rc" = 3 ] && [ "$out" = "$want" ]; then echo "PASS exit 3 with the identity line"; else echo "FAIL exit: rc=$rc out=$out"; fail=1; fi
out=$(cd "$work" && env -u CAPSTONE_EXEC_DIAGNOSTICS sh "$vexec" three.dom 2>&1)
rc=$?
if [ "$rc" = 3 ] && [ -z "$out" ]; then echo "PASS control: no diagnostics, exit 3, no line"; else echo "FAIL control: rc=$rc out=$out"; fail=1; fi

sh "$vexec" --platform > /dev/null 2>&1
rc=$?
if [ "$rc" = 1 ]; then echo "PASS control: --platform refuses without dom_binfmt"; else echo "FAIL platform: rc=$rc"; fail=1; fi
echo 7 > "$work/sublet/parameters/table_growths"
out=$(sh "$vexec" --stats)
want='{"version":2,"route":"trap","sublet.dom_last_base":274877906944,"sublet.table_growths":7}'
case $out in
    "$want") echo "PASS --stats" ;;
    *) echo "FAIL --stats: $out"; fail=1 ;;
esac
out=$(cd "$work" && sh "$vexec" kfault.dom 2>&1)
rc=$?
want="capstone-exec: kernel fault cause=25 address=0x4000002000 nr=63 image=kfault.dom
capstone-exec: exit image=./kfault.dom sha256=$(sha256sum "$work/kfault.dom" | cut -d' ' -f1)"
if [ "$rc" = 0 ] && [ "$out" = "$want" ]; then echo "PASS kernel fault line, exit 0, other pid ignored"; else echo "FAIL kernel fault: rc=$rc out=$out"; fail=1; fi

rm -f "$work/record"
out=$(cd "$work" && CAPSTONE_FAULT_RECORD="$work/record" sh "$vexec" ill.dom 2>&1 | grep "capstone-exec")
rc=$(cd "$work" && sh "$vexec" ill.dom > /dev/null 2>&1; echo $?)
want="capstone-exec: unhandled signal=4 code=0x1 cause=2 pc=0x3ff7f4fadc address=0x3ff7f4fadc ra=0x3ff7f3fac4 image=./ill.dom"
if [ "$rc" = 132 ] && [ "$out" = "$want" ] && [ ! -s "$work/record" ]; then echo "PASS unhandled signal line, no record, SIGILL"; else echo "FAIL unhandled: rc=$rc out=$out record=$(cat "$work/record" 2>/dev/null)"; fail=1; fi
exit $fail
