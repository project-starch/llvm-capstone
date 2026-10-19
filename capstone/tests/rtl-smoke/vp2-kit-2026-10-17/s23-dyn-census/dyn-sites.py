import re, subprocess, sys, collections
OBJ=sys.argv[1]
OPS={0x03:'DELIN',0x51:'CDERIVE',0x52:'CREVOKE'}
for img in sys.argv[2:]:
    out=subprocess.run([OBJ,'-d',img],capture_output=True,text=True).stdout
    fn=None; c=collections.Counter()
    for l in out.splitlines():
        m=re.match(r'^[0-9a-f]+ <(.+)>:$',l)
        if m: fn=m.group(1); continue
        m=re.match(r'^\s*[0-9a-f]+:\s+([0-9a-f]{2}) ([0-9a-f]{2}) ([0-9a-f]{2}) ([0-9a-f]{2})\s',l)
        if not m: continue
        w=int(''.join(reversed(m.groups())),16)
        if w&0x7f==0x5b and (w>>12)&7==1 and (w>>25) in OPS: c[(OPS[w>>25],fn)]+=1
    print('==',img); [print('  ',k[0],k[1],v) for k,v in sorted(c.items())]
