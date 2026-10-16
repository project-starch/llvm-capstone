"""Independent audit of mallocng under option (iii): (a) 7 table edits, (b) skip the 7 bad classes, (b') keep the table,
shift each object to a granule-aligned offset inside its slot. Uniform over requests n in [1, MMAP_THRESHOLD)."""
import sys
import importlib.util, subprocess, tempfile, os
# bounds-model.py at the bitstream's revision, read from capstone-ariane by revision so it cannot drift.
_src = subprocess.run(["git", "-C", os.environ.get("CAPSTONE_ARIANE", "capstone/capstone-ariane"), "show",
                       "d93a8d66b:verif/tests/custom/capstone/bounds-model.py"], capture_output=True, text=True, check=True).stdout
_f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); _f.write(_src); _f.close()
_spec = importlib.util.spec_from_file_location("bounds_model", _f.name); bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)
UNIT, IB, MMAP = 16, 4, 131052
UP = [1,2,3,4,5,6,7,8,9,10,12,15,18,20,25,31,36,42,50,63,72,84,102,127,146,170,204,255,292,340,409,511,
      584,682,818,1023,1169,1364,1637,2047,2340,2730,3276,4095,4680,5460,6552,8191]
ED = list(UP)
for i,v in ((36,1170),(38,1638),(39,2048),(41,2732),(43,4096),(45,5464),(47,8192)): ED[i]=v
def clz32(x): return 32 - x.bit_length()
def size_to_class(n, sc):        # musl 1.2.5 meta.h:229-238 (archive sha256 a9a118bb...), verbatim logic
    n = (n+IB-1)>>4
    if n < 10: return n
    n += 1
    i = (28-clz32(n))*4 + 8
    if n > sc[i+1]: i += 2
    if n > sc[i]: i += 1
    return i
def gran(n):                      # granule of a length at an aligned base (no renormalisation can fire there)
    lz=63
    while lz>12 and ((n>>lz)&1)==0: lz-=1
    E=lz-12
    return 1 if (E==0 and not (n>>12)&1) else 1<<(E+3)
def rr(n): g=gran(n); return -(-n//g)*g
# Positive control on the model itself: gran()/rr() must agree with bounds_model on aligned bases.
bad=0
for n in list(range(4090,4200))+[13073,21793,65520,100000,131000]:
    g=gran(n); base=0x4000_0000; r=rr(n)
    b,t=bm.decompress(*bm.compress(base,base+r)[:3],base)
    if (b,t)!=(base,base+r): bad+=1
assert bad==0, bad
def place(n, table, mode):
    """-> (class, stride) the request ends in, or ('mmap', bytes)."""
    if n >= MMAP: return ('mmap', None)
    if n < 4096 and mode != 'baseline': mode = 'exact'       # granule 1: nothing to do
    m = n
    while True:
        if m >= MMAP: return ('mmap', None)
        sc = size_to_class(m, table); stride = table[sc]*UNIT
        if mode in ('baseline', 'exact'): return (sc, stride)
        g = gran(n); need = rr(n)
        good = stride % g == 0
        if mode == 'a':  ok = need <= stride - IB                                   # strides edited to be good
        elif mode == 'b': ok = good and need <= stride - IB                         # skip bad classes
        elif mode == "b'": ok = need + (0 if good else g - UNIT) <= stride - IB      # shift within the slot
        if ok: return (sc, stride)
        m = table[sc]*UNIT - IB + 1                      # the wrapper passes a request just past this class
def run(table, mode):
    tot=prom=mm=0; worst=(0,0)
    base_tot=0
    for n in range(1, MMAP):
        b = place(n, UP, 'baseline'); p = place(n, table, mode)
        bs = b[1]; ps = p[1]
        if p[0]=='mmap': mm+=1; ps = -(-(rr(n)+UNIT)//4096)*4096   # a page-rounded mapping
        base_tot += bs; tot += ps
        if p[0] != b[0]:
            prom+=1; inc=(ps-bs)/bs
            if inc>worst[0]: worst=(inc,n)
    return prom, mm, (tot-base_tot)/base_tot*100, worst
for name, table, mode in (("(a) 7 table edits", ED, 'a'), ("(b) skip the 7 bad classes", UP, 'b'), ("(b') aligned offset in the slot", UP, "b'")):
    prom, mm, pct, worst = run(table, mode)
    print(f"{name:34s} requests whose class differs from upstream {prom:6d} ({prom/(MMAP-1)*100:5.2f}%), onto mmap {mm:6d}, slot bytes {pct:+6.3f}% vs upstream, worst +{worst[0]*100:.1f}% at n={worst[1]}")
