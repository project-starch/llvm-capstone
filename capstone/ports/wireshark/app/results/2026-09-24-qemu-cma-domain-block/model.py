"""Model of the domain block's sizing (modcapstone capstone.c) and the monitor's split
(sbi_capstone.c create_domain), for a domain that DECLARES its data (.capstone_domreq).

module  (capstone.c:164-169 @7440cfc): tot = code_len + SLACK + data; pages = ceil(tot/4K);
        block = 2^ceil(log2(pages)) pages
monitor (sbi_capstone.c:292-293, the 16-rounding, :918-933): code = roundup16(code_len);
        repr_len = block - code - 1536; hb = floor(log2 repr_len); e = max(0, hb-12);
        gran = 2^(e+3); split = roundup(code, gran); data_off = roundup(1536, gran);
        dom_data = block - split - data_off
A domain is SHORT when dom_data < data (what it declared)."""
import random, sys
PAGE, MDATA, CEIL = 4096, 1536, 4 << 20
def ru(x, g): return (x + g - 1) // g * g
def block_of(tot):
    pages = (tot - 1) // PAGE + 1
    log2 = 0 if pages == 1 else (pages - 1).bit_length()
    return (1 << log2) * PAGE
def gran_of(block, code_len):
    repr_len = block - ru(code_len, 16) - MDATA
    hb = repr_len.bit_length() - 1
    return 1 << (max(0, hb - 12) + 3)
def dom_data(block, code_len):
    g = gran_of(block, code_len)
    return block - ru(ru(code_len, 16), g) - ru(MDATA, g)
def size_old(code_len, data):
    return block_of(code_len + 8192 + data)
def size_new(code_len, data):
    # the fix: size as before, then check the monitor's own split and double until it fits
    block = block_of(code_len + 8192 + data)
    while dom_data(block, code_len) < data:
        block *= 2
    return block
def sweep(sizer, cases):
    short = [(c, d, sizer(c, d)) for c, d in cases if dom_data(sizer(c, d), c) < d]
    return short
random.seed(1)
cases = []
for p in range(16, 29):                      # blocks from 64 KiB to 256 MiB
    B = 1 << p
    for code_frac in (0.05, 0.3, 0.5, 0.7, 0.9):
        code = int(B * code_frac) | random.randrange(1, 16)
        for under in list(range(0, 300 * 1024, 4096)) + [8192, 8191, 8193]:
            d = B - code - 8192 - under       # tot lands just under the power of two
            if d > 0: cases.append((code, d))
for _ in range(200000):
    code = random.randrange(4096, 60 << 20); d = random.randrange(4096, 60 << 20)
    cases.append((code, d))
old = sweep(size_old, cases); new = sweep(size_new, cases)
small_old = [x for x in old if x[2] <= CEIL]
print(f"cases {len(cases)}")
print(f"old rule: SHORT in {len(old)} cases; of those with a block <= 4 MiB: {len(small_old)}")
if old:
    c, d, b = max(old, key=lambda x: d_short(x) if False else x[1] - dom_data(x[2], x[0]))
    print(f"  worst: code {c} data {d} block {b>>20} MiB, dom_data {dom_data(b,c)} = short by {d - dom_data(b,c)}")
print(f"new rule: SHORT in {len(new)} cases")
same_small = sum(1 for c, d in cases if size_old(c, d) <= CEIL and size_old(c, d) != size_new(c, d))
print(f"blocks <= 4 MiB whose size the fix changes: {same_small}")
grown = sum(1 for c, d in cases if size_old(c, d) != size_new(c, d))
print(f"cases the fix gives a larger block: {grown}")
