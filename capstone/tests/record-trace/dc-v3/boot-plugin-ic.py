#!/usr/bin/env python3
"""boot-plugin.py QEMU FW IMAGE INITRAMFS LOG SSTC(on|off) (nested lane, 2026-11-26): runtime's trapk/boot-long.py
(copied, never edited there) with two changes for a TCG plugin that writes its file at QEMU's exit:
  - $BOOT_PLUGIN, when set, is passed as -plugin <value>;
  - after 'TRAPTEST: done' it waits up to $BOOT_EXIT_S seconds (default 300) for QEMU to exit by itself (init's
    poweroff), and only then terminates it. It prints 'BOOT: qemu exited' or 'BOOT: qemu terminated after done'.
boot-plugin-ic.py (2026-11-27) adds $BOOT_ICOUNT, passed as -icount <value>.
Everything else (the command line, the TRAPTEST/panic/sublet line collection) is boot-long.py's."""
import os, sys, pexpect
q, fw, image, initrd, log, sstc = sys.argv[1:7]
cpu = ('rv64,sstc=%s,x-capstone-trap-route=true,x-capstone-supervisor-refs=true,'
       'x-capstone-raw-moves=true' % ('true' if sstc == 'on' else 'false'))
cmd = [q, '-M', 'virt', '-m', os.environ.get('BOOT_MEM', '512M'), '-smp', '1', '-nographic', '-cpu', cpu, '-bios', fw,
       '-kernel', image, '-initrd', initrd,
       '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1 sublet.dom_binfmt=1 ' +
       os.environ.get('TRAPTEST_APPEND', '')]
if os.environ.get('BOOT_ICOUNT'):
    cmd[1:1] = ['-icount', os.environ['BOOT_ICOUNT']]
if os.environ.get('BOOT_PLUGIN'):
    cmd[1:1] = ['-plugin', os.environ['BOOT_PLUGIN']]
out = open(log, 'w')
out.write('# ' + ' '.join(cmd) + '\n')
p = pexpect.spawn(cmd[0], cmd[1:], encoding='utf-8', codec_errors='replace', timeout=int(os.environ.get('BOOT_SILENCE_S', '300')))
p.logfile_read = out
lines = []
done = False
while True:
    i = p.expect([r'TRAPTEST: [^\r\n]*\r?\n', r'Kernel panic[^\r\n]*\r?\n', r'sublet: [^\r\n]*\r?\n', pexpect.EOF, pexpect.TIMEOUT])
    if i in (0, 1, 2):
        lines.append(p.after.strip())
        if i == 1 or p.after.strip() == 'TRAPTEST: done':
            done = p.after.strip() == 'TRAPTEST: done'
            try:
                p.expect(pexpect.EOF, timeout=int(os.environ.get('BOOT_EXIT_S', '300')) if done else 10)
                lines.append('BOOT: qemu exited')
            except pexpect.TIMEOUT:
                lines.append('BOOT: qemu terminated after done' if done else 'BOOT: qemu terminated after panic')
            break
    else:
        lines.append('BOOT: ' + ('EOF' if i == 3 else 'TIMEOUT'))
        break
p.terminate(force=True)
out.close()
print('\n'.join(lines))
