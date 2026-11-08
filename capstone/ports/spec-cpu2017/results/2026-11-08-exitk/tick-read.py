#!/usr/bin/env python3
"""tick-read.py <spec.txt>...: the readings PREREG-nohz.md and PREREG-hbfix.md grade, from one stage transcript each.
- @@TL: the tick device's event_handler (and nohz_mode when printed);
- @@TK0/@@TK1: /proc/stat cpu ticks accounted over the command, against (uptime delta x HZ=100); intr and ctxt deltas;
- @@CPU0/@@CPU1 (the cpusnap stages): the same cpu line without uptime;
- @@PST0/@@PST1: the largest cutime+cstime deltas (the heartbeat subshell's children, the stage shell's children);
- the module's last `capstone_vm: exits` and `bench` lines: kcyc - rcyc, ucyc, steps, inside cycles.
A transcript with none of these is an error, not a zero."""
import re, sys
def read(path):
    t = open(path, 'rb').read().decode('latin-1')
    out, found = [], 0
    m = re.search(r'@@TL ([^\n]*)', t)
    if m: found += 1; out.append('TL %s' % ' '.join(m.group(1).split()))
    tk = {k: v for k, v in re.findall(r'@@TK([01]) cpu ([^\n]*)', t)}
    if len(tk) == 2:
        found += 1
        def f(s):
            n = s.split(); cpu = [int(x) for x in n[:10]]
            kv = dict(x.split('=') for x in n[10:])
            return cpu, kv
        (c0, k0), (c1, k1) = f(tk['0']), f(tk['1'])
        ticks = sum(c1) - sum(c0); wall = float(k1['up']) - float(k0['up'])
        out.append('TK ticks %d over %.2f s = %.1f %% of wall x 100; user %d system %d idle %d; intr %d ctxt %d' % (
            ticks, wall, 100.0 * ticks / (wall * 100), c1[0] - c0[0], c1[2] - c0[2], c1[3] - c0[3],
            int(k1['intr']) - int(k0['intr']), int(k1['ctxt']) - int(k0['ctxt'])))
    cs = {k: v for k, v in re.findall(r'@@CPU([01]) cpu ([^\n]*)', t)}
    if len(cs) == 2:
        found += 1
        c0 = [int(x) for x in cs['0'].split()]; c1 = [int(x) for x in cs['1'].split()]
        out.append('CPU ticks %d; user %d system %d idle %d softirq %d (cpusnap: no uptime of its own)' % (
            sum(c1) - sum(c0), c1[0] - c0[0], c1[2] - c0[2], c1[3] - c0[3], c1[6] - c0[6]))
    p = {0: {}, 1: {}}
    for w, pid, comm, rest in re.findall(r'@@PST([01]) (\d+) \((.*?)\) \S ([^\n]*)', t):
        f = rest.split(); p[int(w)][(pid, comm)] = (int(f[0]), int(f[10]), int(f[11]), int(f[12]), int(f[13]))
    if p[0] and p[1]:
        found += 1
        rows = []
        for k in set(p[0]) & set(p[1]):
            a, b = p[0][k], p[1][k]
            rows.append((b[3] + b[4] - a[3] - a[4], b[1] + b[2] - a[1] - a[2], k, a[0]))
        for ch, own, (pid, comm), ppid in sorted(rows, reverse=True)[:4]:
            out.append('PST pid %s (%s) ppid %d: children %d ticks, own %d ticks' % (pid, comm, ppid, ch, own))
    e = re.findall(r'capstone_vm: exits steps=(\d+) kcyc=(\d+) rcyc=(\d+) ucyc=(\d+) ocyc=(\d+) ev=([\d,]+)', t)
    b = re.findall(r'capstone_vm: bench cycles=(\d+) instret=(\d+) entries=(\d+)', t)
    if e and b:
        found += 1
        st, k, r, u, o = map(int, e[-1][:5]); cyc, ins, ent = map(int, b[-1])
        out.append('EXITS steps %d kcyc-rcyc %.3f G ucyc %.3f G rcyc %.3f G ev %s; inside %.3f G cycles, instret %d, entries %d' % (
            st, (k - r) / 1e9, u / 1e9, r / 1e9, e[-1][5], cyc / 1e9, ins, ent))
    u = re.findall(r'up0=([\d.]+) up1=([\d.]+)', t)
    if u: out.append('RUN %.1f s' % (float(u[-1][1]) - float(u[-1][0])))
    if not found:
        sys.exit('NO DATA: no @@TL, @@TK, @@PST or capstone_vm exits lines in %s' % path)
    return out
if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit('usage: tick-read.py <spec.txt>...')
    for a in sys.argv[1:]:
        print('==', a)
        for l in read(a): print('  ' + l)
