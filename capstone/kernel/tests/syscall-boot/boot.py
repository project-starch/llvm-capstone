#!/usr/bin/env python3
"""boot.py QEMU FW IMAGE INITRAMFS LOG: boot the test kernel on QEMU virt (one hart) with the
Sublet trap-route CPU, print the S5TEST and S5 lines and a panic."""
import sys, pexpect
q, fw, image, initrd, log = sys.argv[1:6]
cpu = 'rv64,sstc=false,x-capstone-trap-route=true,x-capstone-supervisor-refs=true,x-capstone-raw-moves=true'
cmd = [q, '-M', 'virt', '-m', '512M', '-smp', '1', '-nographic', '-cpu', cpu, '-bios', fw,
       '-kernel', image, '-initrd', initrd,
       '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1']
out = open(log, 'w')
out.write('# ' + ' '.join(cmd) + '\n')
p = pexpect.spawn(cmd[0], cmd[1:], encoding='utf-8', codec_errors='replace', timeout=300)
p.logfile_read = out
lines = []
while True:
    # Each pattern ends at the line end, so a line arriving in pieces is not cut short.
    i = p.expect([r'S5TEST: [^\r\n]*\r*\n', r'S5: [^\r\n]*\r*\n', r's5 write ok\r*\n',
                  r'Kernel panic[^\r\n]*\r*\n', r'Oops[^\r\n]*\r*\n', pexpect.EOF, pexpect.TIMEOUT])
    if i < 5:
        lines.append(p.after.strip())
        if i == 3 or p.after.strip() == 'S5TEST: done':
            break
    else:
        lines.append('BOOT: ' + ('EOF' if i == 5 else 'TIMEOUT'))
        break
p.terminate(force=True)
out.close()
print('\n'.join(lines))
