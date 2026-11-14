"""Check the built syscall boundary, not just its assembly source."""
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
assembly = (root / 'syscall.dis').read_text()
stub = assembly.split('<__capstone_linux_clone>:', 1)[0]
assert re.search(r'\bmv\s+a7,\s*a0', stub), 'Linux number must reach a7'
assert len(re.findall(r'\becall\b', stub)) == 1
assert not re.search(r'\bli\s+a7,', stub), 'service selector survived'
for dst, src in zip(range(6), range(1, 7)):
    assert re.search(r'\bmovc\s+a%d,\s*a%d' % (dst, src), stub), (dst, src)
image = (root / 'smoke.dis').read_text()
for name in ('__libc_start_main', '__capstone_linux_start', '__mmap', '__munmap'):
    assert '<' + name + '>:' in image or '<' + name.lstrip('_') + '>:' in image, name
assert 'domain_main' not in image
assert '__capstone_vm_' not in image
print('PASS: plain ecall, number in a7, six MOVC arguments, musl startup, no launcher stubs')

probes = (root / 'syscall-shape.dis').read_text()
for name, nr in {'read': 63, 'write': 64, 'openat': 56, 'close': 57,
                 'mmap': 222, 'munmap': 215, 'mprotect': 226,
                 'exit': 93, 'exit_group': 94, 'getpid': 172,
                 'clock_gettime': 113, 'rt_sigaction': 134,
                 'rt_sigprocmask': 135, 'sigaltstack': 132}.items():
    body = probes.split('<probe_' + name + '>:', 1)[1]
    body = re.split(r'<probe_\w+>:', body)[0]
    numbers = re.findall(r'\bli\s+a0,\s*(0x[0-9a-f]+|[0-9]+)', body)
    assert nr in [int(n, 0) for n in numbers], (name, nr, numbers)
    print('PASS: Linux %s = %d' % (name, nr))

clone = assembly.split('<__capstone_linux_clone>:', 1)[1].split('<__unmapself>:', 1)[0]
for dst, src in ((2, 4), (3, 5), (4, 6)):
    assert re.search(r'\bmovc\s+a%d,\s*a%d' % (dst, src), clone)
for insn in ('stc', 'ldc', 'cincoffsetimm'):
    assert re.search(r'\b' + insn + r'\b', clone), insn
assert re.search(r'\bli\s+a7,\s*(220|0xdc)\b', clone)
threads = (root / 'threads-smoke.dis').read_text()
for name in ('__capstone_linux_clone', 'pthread_create', 'pthread_join'):
    assert '<' + name + '>:' in threads or '<__' + name + '>:' in threads, name
print('PASS: clone keeps stack, TLS and tid references, pthread smoke links')

jumps = (root / 'setjmp-smoke.dis').read_text()
# Local labels do not delimit a function in llvm-objdump output.
def function(text, name, next_name):
    return text.split('<' + name + '>:', 1)[1].split('<' + next_name + '>:', 1)[0]

for name, end in (('setjmp', 'sigsetjmp'), ('sigsetjmp', 'siglongjmp')):
    body = function(jumps, name, end)
    for i, reg in enumerate(['ra', *('s%d' % n for n in range(12)), 'sp']):
        assert re.search(r'\bstc\s+' + reg + r',\s*0x%x\(a0\)' % (i * 16), body), (name, reg)
body = function(jumps, 'longjmp', 'memcpy')
for i, reg in enumerate(['ra', *('s%d' % n for n in range(12)), 'sp']):
    assert re.search(r'\bldc\s+' + reg + r',\s*0x%x\(a0\)' % (i * 16), body), reg
for name, end in (('sigsetjmp', 'siglongjmp'), ('siglongjmp', 'longjmp')):
    body = function(jumps, name, end)
    assert re.search(r'\bli\s+a7,\s*0x87', body), name
    assert re.search(r'\bli\s+a3,\s*0x8', body), name
    assert len(re.findall(r'\becall\b', body)) == 1, name
for image in (jumps, (root / 'signal-smoke.dis').read_text()):
    for forbidden in ('__capstone_sigsetjmp_save', '__capstone_longjmp_reap', '__capstone_signal', '__capstone_vm_'):
        assert forbidden not in image, forbidden
assert re.search(r'\blcc\s+t0,\s*s2,', jumps), 'missing restored-reference tag check'
print('PASS: fourteen tagged jump slots, mask ecalls, restored-reference tag check, no signal transport')
