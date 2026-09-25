#!/usr/bin/env python3
"""Split libc_test_host's stdout into the domain's own two streams (host/run-qemu.sh's oracle).

    domain-stdout.py <host-stdout> <out> [<expected-stderr>]

The host's stdout is, in order:
  1. the loader's banner, create_dom's report on the image: "Ok, good file." through
     "Loadable size = <n>";
  2. the domain's writes to fd 1 and fd 2, interleaved as written (the runtime sends both as
     WRITE_STDOUT). domain_entry.c's echoes of argv and the environment are the first of them;
  3. one "TSAPP-HEAP ..." line, written by the exit hook (src/tsapp-heap.c) after exit() has
     flushed stdout, so after everything tshark wrote. It carries the unserved syscalls and
     whether fd 1 was still open, and its list is the one reported;
  4. if the program made any syscall the runtime does not serve, one
     "capstone-domain: UNSERVED syscalls: <nr>[x<times>] ..." line, which domain_main
     (runtime/hostcall.c, hc_report_unserved) writes after the exit hook has returned;
  5. one "LT-RESULT ..." line, on a line of its own because the host prints it after the
     domain has returned.
This removes 1, the "domain_entry: " lines of 2, and 3 to 5, and nothing else. Any other shape is
an error (exit 1, reason on <out>.err), never a pass-through. The unserved list is a finding, not
noise: it is printed as `unserved: <list>` or `unserved: none`.

The echoes are checked, not only removed: with DOMAIN_ARGV and DOMAIN_ENV set to the files the
runner gave the guest, the echoed argv and environment must be exactly their lines, in order, or it
is an error (a run that did not get its command line is not a run of that command).

With <expected-stderr>: what is left must BEGIN with exactly those bytes, which are then removed,
and <out> is the rest. tshark's stderr on these runs is all written during registration, before
its first write to stdout (patch 0002's notices for dissector tables the whitelist leaves out), so
the two streams do not interleave. Prints `stderr MATCH` or `stderr DIFFERS`; on DIFFERS nothing
is removed, so the stdout comparison that follows cannot match either. An EMPTY <expected-stderr>
is an error: every output begins with zero bytes, so it would match anything.
"""
import os, re, sys

src, dst = sys.argv[1], sys.argv[2]
want_err = open(sys.argv[3], 'rb').read() if len(sys.argv) > 3 else None
lines = open(src, 'rb').read().split(b'\n')

def fail(msg):
    open(dst + '.err', 'w').write(msg + '\n')
    sys.exit(1)

if want_err is not None and not want_err:
    fail('the expected stderr is empty, which would match anything')

if not lines or lines[0] != b'Ok, good file.':
    fail('no loader banner at the top')
end = next((i for i, l in enumerate(lines) if re.fullmatch(rb'Loadable size = [0-9]+', l)), None)
if end is None or end > 20:
    fail('the loader banner does not end in "Loadable size = <n>"')
body = lines[end + 1:]
# The data ends with "\n", so the last element is empty; the LT-RESULT line is the one before it.
if len(body) < 2 or body[-1] != b'' or not body[-2].startswith(b'LT-RESULT '):
    fail('the output does not end with an LT-RESULT line')
body = body[:-2]
unserved = 'none'
if body and body[-1].startswith(b'capstone-domain: UNSERVED syscalls:'):
    unserved = body[-1][len(b'capstone-domain: UNSERVED syscalls:'):].decode().strip()
    body = body[:-1]
if not body or not body[-1].startswith(b'TSAPP-HEAP '):
    fail('no TSAPP-HEAP line before the LT-RESULT line')
heap = body[-1].decode('latin-1')
body = body[:-1]
# The exit hook's own report (src/tsapp-heap.c), on fd 2: it survives a program that closed fd 1,
# which the runtime's line above does not.
m = re.search(r' unserved=(\S+) stdout=(open|closed)$', heap)
if not m:
    fail('the TSAPP-HEAP line has no unserved=/stdout= fields')
hook_unserved, stdout_state = m.group(1).replace(',', ' '), m.group(2)
echo = re.compile(rb'domain_entry: (argv\[([0-9]+)\]=|env )(.*)')
got_argv, got_env, kept = [], [], []
for l in body:
    m = echo.fullmatch(l)
    if not m:
        kept.append(l)
    elif m.group(2) is not None:
        got_argv.append(m.group(3))
    else:
        got_env.append(m.group(3))
for var, got in (('DOMAIN_ARGV', got_argv), ('DOMAIN_ENV', got_env)):
    if os.environ.get(var):
        want = [l for l in open(os.environ[var], 'rb').read().split(b'\n') if l]
        if got != want:
            fail(f'{var}: the domain echoed {got!r}, the runner wrote {want!r}')
rest = b''.join(l + b'\n' for l in kept)
verdict = []
if want_err is not None:
    if rest.startswith(want_err):
        rest = rest[len(want_err):]
        verdict.append('stderr MATCH')
    else:
        verdict.append('stderr DIFFERS')
verdict.append('unserved: ' + hook_unserved)
if unserved != 'none' and unserved != hook_unserved:
    verdict.append('runtime line: ' + unserved)
if stdout_state != 'open':
    verdict.append('fd 1 CLOSED at exit')
print(', '.join(verdict))
open(dst, 'wb').write(rest)
