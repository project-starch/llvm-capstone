#!/usr/bin/env python3
"""Parse per-context capstone_vm lines (bench, exits, hpm, exhpm, teardown) from a board spec.txt."""
import re, sys, json
def parse(path):
    txt = open(path, 'rb').read().decode('latin-1')
    ctx, cur = [], {}
    for line in txt.splitlines():
        m = re.search(r'\[\s*([\d.]+)\] capstone_vm: (\w+) (.*)', line)
        if not m: continue
        t, kind, rest = float(m.group(1)), m.group(2), m.group(3)
        kv = dict(re.findall(r'(\w+)=([^\s]+)', rest))
        if kind == 'teardown':
            cur = {'t_teardown': t, 'td_pages': int(kv['pages']), 'td_ns': int(kv['ns'])}
        elif kind == 'bench':
            cur.update(t=t, cyc=int(kv['cycles']), ins=int(kv['instret']), entries=int(kv['entries']))
        elif kind == 'exits':
            ev = [int(x) for x in kv['ev'].split(',')]
            cur.update(steps=int(kv['steps']), kcyc=int(kv['kcyc']), rcyc=int(kv['rcyc']),
                       ucyc=int(kv['ucyc']), ocyc=int(kv['ocyc']), ev=ev)
        elif kind == 'hpm':
            cur.update(hpm=[int(kv['hpm%d' % i]) for i in range(3, 9)])
        elif kind == 'exhpm':
            cur.update(k_in=int(kv['k_in']), k=[int(x) for x in kv['k'].split(',')],
                       u_in=int(kv['u_in']), u=[int(x) for x in kv['u'].split(',')])
            ctx.append(cur); cur = {}
            continue
        if kind == 'hpm' and 'exhpm' not in rest:
            # exhpm (if any) follows hpm; flush at next teardown otherwise
            pass
        if kind == 'hpm':
            cur['_done'] = True
    return ctx
def parse_all(path):
    txt = open(path, 'rb').read().decode('latin-1')
    out, cur = [], None
    for line in txt.splitlines():
        m = re.search(r'\[\s*([\d.]+)\] capstone_vm: (\w+) (.*)', line)
        if not m: continue
        t, kind, rest = float(m.group(1)), m.group(2), m.group(3)
        kv = dict(re.findall(r'(\w+)=([^\s]+)', rest))
        if kind == 'teardown':
            cur = {'t_teardown': t, 'td_pages': int(kv['pages']), 'td_ns': int(kv['ns'])}; out.append(cur)
        elif cur is None: continue
        elif kind == 'bench': cur.update(t=t, cyc=int(kv['cycles']), ins=int(kv['instret']), entries=int(kv['entries']))
        elif kind == 'exits':
            cur.update(steps=int(kv['steps']), kcyc=int(kv['kcyc']), rcyc=int(kv['rcyc']), ucyc=int(kv['ucyc']),
                       ocyc=int(kv['ocyc']), ev=[int(x) for x in kv['ev'].split(',')])
        elif kind == 'hpm': cur.update(hpm=[int(kv['hpm%d' % i]) for i in range(3, 9)])
        elif kind == 'exhpm':
            cur.update(k_in=int(kv['k_in']), k=[int(x) for x in kv['k'].split(',')],
                       u_in=int(kv['u_in']), u=[int(x) for x in kv['u'].split(',')])
    out = [c for c in out if 'steps' in c]
    if not out:
        sys.exit('NO DATA: no capstone_vm exits lines in %s' % path)
    return out
if __name__ == '__main__':
    json.dump(parse_all(sys.argv[1]), sys.stdout)
