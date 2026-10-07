#!/usr/bin/env python3
"""Log in on the guest serial console and install an SSH key for root."""
import sys
import time
import pexpect

sock, pubkey = sys.argv[1], sys.argv[2]
key = open(pubkey).read().strip()
for _ in range(60):
    child = pexpect.spawn('socat', ['-', 'UNIX-CONNECT:' + sock], encoding='utf-8', timeout=1800)
    if child.expect([pexpect.EOF, r'.'], timeout=2) == 1:
        break
    time.sleep(1)
child.logfile_read = sys.stdout
child.expect('login:')
child.sendline('root')
child.expect(r'# $')
for line in ('mkdir -p /root/.ssh && chmod 700 /root/.ssh',
             f"echo '{key}' >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys",
             "grep -q '^PermitRootLogin' /etc/ssh/sshd_config || echo 'PermitRootLogin without-password' >> /etc/ssh/sshd_config",
             'service sshd onerestart >/dev/null 2>&1 || service sshd onestart',
             'echo MQ-READY'):
    child.sendline(line)
    child.expect(r'# $', timeout=600)
child.close()
