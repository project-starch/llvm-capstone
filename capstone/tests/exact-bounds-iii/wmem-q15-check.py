#!/usr/bin/env python3
"""Check wmem patch 0001's Q-15 helpers against capstone-qemu's COMPACT encoder, from the compiled header.

Usage: wmem-q15-check.py <patched wsutil/wmem/wmem_allocator.h> [N]   (exit 0 = all checks hold and both controls fire)

The helpers between `#define WMEM_SUBLET_Q15_ALIGN` and the CREVOKE release are cut out of the patched header verbatim and
compiled natively, with the capability builtins reduced to addresses and CDERIVE recorded rather than executed. N random
issues (sizes 1..70,000, 1..40 MiB and 4,000..44,000; chunk data 16-aligned inside a 64 MiB buffer aligned to 2^26, above
every granule here) go through the real wmem_sublet_issue_at, which writes its offset word into real memory. Each child
[at, at+len) is judged by qemu_compact_repr.py (a port of cap_compress at b290f5656600):
  exact; tail pad only (len >= size, pad below the granule); inside the bytes the allocator served
  (wmem_sublet_request); a placed child at least WMEM_SUBLET_Q15_SLACK past the data with its offset stored below it;
  and the jumbo length wmem_sublet_block_size(request + 80) exact at a granule-aligned base.
Controls, both must fire: the unrepaired child [ptr, ptr+size) and the unrounded jumbo length, refused by the encoder.
"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qemu_compact_repr import representable as _rep
REPR_H = os.path.join(HERE, '../../runtime/include/capstone/representable.h')

def representable(base, size):
    return _rep(base, size)[0]          # (ok, detail): the tuple itself is always truthy

def granule(n):
    return 1 if n < 4096 else 1 << (n.bit_length() - 1 - 9)

HARNESS = r'''
int main(int argc, char **argv) {
  unsigned long seed = 12345, N = strtoul(argv[1], 0, 0);
  void *bufv; if (posix_memalign(&bufv, 1UL << 26, 1UL << 26)) return 2;
  uint8_t *buf = bufv;
  wmem_sublet_block_t s = { 0, 0 };
  for (unsigned long i = 0; i < N; i++) {
    seed = seed * 6364136223846793005UL + 1442695040888963407UL;
    unsigned long r = seed >> 11;
    size_t size = (i % 3 == 0) ? 1 + r % 70000 : (i % 3 == 1) ? 1 + r % (40UL << 20) : 4000 + r % 40000;
    size_t req = wmem_sublet_request(size);
    if (req > (1UL << 26) - 64) continue;
    uintptr_t off = ((r >> 3) % ((1UL << 26) - req)) & ~15UL;
    uint8_t *ptr = buf + off, *at;
    wmem_sublet_issue_at(&s, ptr, size, &at, 0);
    unsigned long stored = (at != ptr) ? *(uint64_t *)(at - 8) : 0;
    printf("%zu %zu %lu %lu %lu %lu %lu\n", size, req, (unsigned long)(uintptr_t)ptr, (unsigned long)(uintptr_t)at,
           last_len, stored, (unsigned long)wmem_sublet_block_size(req + 80));
  }
  return 0;
}
'''

def main():
    hdr, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    s = open(hdr).read()
    a, b = s.index('#define WMEM_SUBLET_Q15_ALIGN'), s.index("/* End the caller's object (CREVOKE)")
    pre = ('#include <stdint.h>\n#include <stddef.h>\n#include <stdbool.h>\n#include <string.h>\n#include <stdio.h>\n'
           '#include <stdlib.h>\n#include "%s"\n'
           'typedef struct { void *life, *generation; } wmem_sublet_block_t;\n'
           '#define __builtin_capstone_cap_get_cursor(p) ((unsigned long)(uintptr_t)(p))\n'
           '#define __builtin_capstone_cap_get_base(p) ((unsigned long)(uintptr_t)(p))\n'
           'static unsigned long last_len;\n'
           '#define capstone_cap_derive(parent, off, len) (last_len = (len), (void *)((uint8_t *)(parent) + (off)))\n'
           'unsigned long wmem_sublet_q15_chunks, wmem_sublet_q15_padded, wmem_sublet_q15_pad_total, '
           'wmem_sublet_q15_pad_max;\nvoid wmem_sublet_q15_counts(unsigned long out[4]) { (void)out; }\n'
           % os.path.abspath(REPR_H))
    with tempfile.TemporaryDirectory() as t:
        open(f'{t}/h.c', 'w').write(pre + s[a:b] + HARNESS)
        subprocess.run(['cc', '-O1', '-Wall', '-Werror', '-o', f'{t}/h', f'{t}/h.c'], check=True)
        rows = [list(map(int, l.split())) for l in subprocess.run([f'{t}/h', str(n)], capture_output=True,
                                                                  text=True, check=True).stdout.splitlines()]
    if not rows:
        sys.exit('NO DATA: the harness printed no issues')
    bad = dict(child=0, pad=0, inside=0, slack=0, jumbo=0); placed = big = ctl_child = ctl_jumbo = 0
    base = 1 << 40
    for size, req, ptr, at, ln, stored, J in rows:
        bad['child'] += not representable(at, ln)
        bad['pad'] += not (ln >= size and ln - size < granule(ln))
        bad['inside'] += not (at >= ptr and at + ln <= ptr + req)
        if at != ptr:
            placed += 1
            bad['slack'] += not (at - ptr >= 16 and stored == at - ptr)
        if J > (2 << 20):
            bad['jumbo'] += not representable(base, J)
            ctl_jumbo += not representable(base, req + 80)
        if size >= 4096:
            big += 1
            ctl_child += not representable(ptr, size)
    print(f"issues {len(rows)} placed {placed} failures {bad}")
    print(f"CONTROLS: unrepaired child refused {ctl_child} of {big} (size >= 4096); unrounded jumbo refused {ctl_jumbo}")
    ok = not any(bad.values()) and ctl_child > 0 and ctl_jumbo > 0
    print('WMEM-Q15-CHECK', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

main()
