import sys,re
s=open(sys.argv[1]).read()
s=s.split('__HASH__')[0]
for blk in (r'%ifndef %\{sdk\}.*?%endif\n', r'%ifndef %\{kit\}.*?%endif\n', r'%ifndef %\{arm_label\}.*?%endif\n', r'%ifdef %\{vm_state\}.*?%endif\n'):
    s,n=re.subn(blk,'',s,flags=re.S); assert n==1,blk
pairs=[('label                = %{arm_label}','label                = plain-rv64'),
 ('   CC                     = %{sdk}/capstone-cc','   CC                     = %{plain_cc}'),
 ('   sw_base_ptrsize        = 128-bit capabilities, 64-bit long','   sw_base_ptrsize        = 64-bit'),
 ('   # long is 64 bits; pointers are 128-bit capabilities, which no SPEC data\n   # model names. LP64 is the closest one, and every use of it is a finding.\n','   # Plain riscv64 Linux: LP64, as in the Capstone arms.\n')]
for a,b in pairs:
    assert s.count(a)==1,a; s=s.replace(a,b)
hdr=('# plain-rv64: the Plain baseline of the SPEC train runs. Generated from the kit\'s capstone-virtual.cfg by\n'
     '# spec/plain/mkcfg.py: the same OPTIMIZE, PORTABILITY and src.alt per benchmark; only the compiler (plain-cc:\n'
     '# the same clang, riscv64 Linux musl, static) and the label differ.\n'
     '%ifndef %{plain_cc}\n%   error define plain_cc=<plain-cc>\n%endif\n')
open(sys.argv[2],'w').write(hdr+s)
