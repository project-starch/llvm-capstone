#!/usr/bin/env python3
"""boot.py QEMU FW IMAGE INITRAMFS LOG SSTC(on|off): boot the trap-route test kernel on QEMU virt
(one hart) with the Sublet trap-route CPU, $TRAPTEST_APPEND added to the command line, collect the TRAPTEST lines and a panic, print them."""
import os, sys, pexpect
q, fw, image, initrd, log, sstc = sys.argv[1:7]
cpu = ('rv64,sstc=%s,x-capstone-trap-route=true,x-capstone-supervisor-refs=true,'
       'x-capstone-raw-moves=true' % ('true' if sstc == 'on' else 'false'))
cmd = [q, '-M', 'virt', '-m', '512M', '-smp', '1', '-nographic', '-cpu', cpu, '-bios', fw,
       '-kernel', image, '-initrd', initrd,
       '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1 sublet.dom_binfmt=1 ' +
       os.environ.get('TRAPTEST_APPEND', '')]
out = open(log, 'w')
out.write('# ' + ' '.join(cmd) + '\n')
p = pexpect.spawn(cmd[0], cmd[1:], encoding='utf-8', codec_errors='replace', timeout=300)
p.logfile_read = out
lines = []
while True:
    i = p.expect([r'TRAPTEST: [^\r\n]*\r*\n', r'Kernel panic[^\r\n]*\r*\n', r'sublet: [^\r\n]*\r*\n', pexpect.EOF, pexpect.TIMEOUT])
    if i in (0, 1, 2):
        lines.append(p.after.strip())
        if i == 1 or p.after.strip() == 'TRAPTEST: done':
            if i == 1:
                try:
                    p.expect(pexpect.EOF, timeout=10)
                except pexpect.TIMEOUT:
                    pass
            break
    else:
        lines.append('BOOT: ' + ('EOF' if i == 3 else 'TIMEOUT'))
        break
p.terminate(force=True)
out.close()
print('\n'.join(lines))
