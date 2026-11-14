#!/usr/bin/env python3
"""boot.py QEMU FIRMWARE KERNEL INITRAMFS LOG: run and check one short boot."""
import re
import sys
import pexpect

qemu, firmware, kernel, initramfs, logfile = sys.argv[1:]
command = [qemu, '-M', 'virt', '-m', '512M', '-smp', '1', '-nographic',
           '-cpu', 'rv64,sstc=false,x-capstone-trap-route=true,'
                   'x-capstone-supervisor-refs=true,x-capstone-raw-moves=true',
           '-bios', firmware, '-kernel', kernel, '-initrd', initramfs,
           '-append', 'console=ttyS0 earlycon=sbi rdinit=/init panic=-1 sublet.dom_binfmt=1']
with open(logfile, 'w') as stream:
    child = pexpect.spawn(command[0], command[1:], encoding='utf-8',
                          codec_errors='replace', timeout=90)
    child.logfile_read = stream
    try:
        child.expect(r'TRAPTEST: done\r*\n')
    finally:
        child.terminate(force=True)
text = open(logfile).read()
for name in ['sqlite-request', 'sqlite-block', 'sqlite-page', 'ffmpeg', 'ffmpeg-ref-failure']:
    assert f'TRAPTEST: result {name} 0\n' in text, name
for name in ['sqlite-old', 'ffmpeg-old', 'ffmpeg-old-ref-failure']:
    assert re.search(r'TRAPTEST: result ' + name + r' (134|139)\n', text), name
for name in ['sqlite-request', 'sqlite-block', 'sqlite-page', 'ffmpeg-buffer', 'ffmpeg-refstruct']:
    section = text.split(f'TRAPTEST: stale {name}\n', 1)[1].split('TRAPTEST: stale-result', 1)[0]
    assert 'fault cause=25 ' in section, name
for line in text.splitlines():
    if line.startswith(('TRAPTEST:', 'LIFETIME ')) or 'fault cause=' in line:
        print(line)
print('LIFETIME boot PASS')
