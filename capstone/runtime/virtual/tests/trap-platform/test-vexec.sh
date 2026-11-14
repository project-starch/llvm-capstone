#!/bin/sh
# test-vexec.sh: trap-vexec.sh on the host, with a fake dmesg and parameter directory.
#   1. An image that the "kernel" kills with a Sublet fault: the launcher line, death by SIGSEGV (139).
#   2. Control: an image that exits 3 prints no line and exits 3.
#   3. Control: --platform refuses when dom_binfmt is not Y.
#   4. --stats prints the numeric parameters as JSON and skips the others.
#   5. A kernel access fault of the child: the kernel-fault line, and the child's own exit status 0.
#      Control: another pid's kernel fault prints nothing.
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
chmod +x "$work/bin/dmesg" "$work/fault.dom" "$work/three.dom"
fail=0
export PATH="$work/bin:$PATH" SUBLET_PARAMETERS="$work/sublet/parameters" CAPSTONE_EXEC_DIAGNOSTICS=1

out=$(cd "$work" && sh "$vexec" fault.dom 2>&1 | grep "domain fault")
rc=$(cd "$work" && sh "$vexec" fault.dom > /dev/null 2>&1; echo $?)
want="capstone-exec: domain fault cause=25 pc=0x4000001010 address=0x55 ra=0x0 fp=0x0 entry=0x4000000100 code=0x4000000000-0x4000000000 image=./fault.dom"
if [ "$rc" = 139 ] && [ "$out" = "$want" ]; then echo "PASS fault line and SIGSEGV"; else echo "FAIL fault: rc=$rc out=$out"; fail=1; fi

out=$(cd "$work" && sh "$vexec" three.dom 2>&1)
rc=$?
if [ "$rc" = 3 ] && [ -z "$out" ]; then echo "PASS control: exit 3, no line"; else echo "FAIL control: rc=$rc out=$out"; fail=1; fi

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
want="capstone-exec: kernel fault cause=25 address=0x4000002000 nr=63 image=kfault.dom"
if [ "$rc" = 0 ] && [ "$out" = "$want" ]; then echo "PASS kernel fault line, exit 0, other pid ignored"; else echo "FAIL kernel fault: rc=$rc out=$out"; fail=1; fi
exit $fail
