#!/usr/bin/env python3
"""read-boardS.py LOG TWINLOG (nested lane, 2026-11-25): a B5b-2 boot of M-16's stress form (make-boardS.py) from its
console log, against the trap-route twin's log of the same stages (carriage returns dropped in both). Reads:
  - the control line "TRAPTEST: control smoke.dom rc=7";
  - per command (u0, u1, p0, p1, p2, those present in the twin's log), in the order the twin ran them:
      RETURNED   rc 0 and every STRESS line equal to the twin's, in order;
      NO RETURN  @@SPEC-BEGIN without @@SPEC-END: the last STRESS line, its fast-allocation count, the twin's next
                 heartbeat (the stop lies between the two), and whether /init's 120 s ticks continued after the last
                 heartbeat (the kernel ran on: the stop is in the program) or stopped (the core or kernel stopped);
                 a "watchdog killed" line from /init after it is reported (the 3,600 s stage bound, not a stop);
      OTHER      any STRESS line differs from the twin's, or rc nonzero;
      NOT RUN    absent from the board's log;
  - the sum of fast allocations issued by the protected commands that returned, plus those completed before a stop.
Exit 0 when the control is met and every command present in the log is RETURNED or NO RETURN; 1 otherwise or when
the log has no command at all."""
import re, sys

def load(p):
    return open(p, errors='replace').read().replace('\r', '')

def commands(t):
    out = {}
    for m in re.finditer(r'^@@SPEC-BEGIN \d+ run (\w+) 0$', t, re.M):
        name, start = m.group(1), m.end()
        e = re.compile(r'^@@SPEC-END \d+ run %s 0 rc=(\d+) up0=([\d.]+) up1=([\d.]+)$' % name, re.M).search(t, start)
        body = t[start:e.start()] if e else t[start:]
        stress = re.findall(r'^STRESS .*$', body, re.M)
        ticks = [float(x) for x in re.findall(r'^TRAPTEST: tick ([\d.]+)$', body, re.M)]
        q = t.find('@@QSTEP-BEGIN', start)  # the next stage (the watchdog's own line can follow this stage's QSTEP-END)
        wd = re.search(r'^TRAPTEST: watchdog killed .*$', t[start:q if q >= 0 else len(t)], re.M)
        out[name] = dict(end=e, stress=stress, body=body, wd=wd.group(0) if wd else None, ticks=ticks)
    return out

def fast(line):
    m = re.search(r'fast=(\d+)', line or '')
    return int(m.group(1)) if m else 0

def main():
    t, tw = load(sys.argv[1]), load(sys.argv[2])
    bad = 0
    ctl = re.search(r'TRAPTEST: control smoke\.dom rc=(\d+)', t)
    print(f'control smoke.dom rc={ctl.group(1) if ctl else None} ({"met" if ctl and ctl.group(1) == "7" else "NOT MET"})')
    bad |= not (ctl and ctl.group(1) == '7')
    B, W = commands(t), commands(tw)
    if not B:
        print('NO COMMAND in the log'); sys.exit(1)
    issued = 0
    for name, w in W.items():
        assert w['end'] and w['end'].group(1) == '0', f'twin {name} did not return rc 0'
        if name not in B:
            print(f'{name}: NOT RUN'); continue
        b = B[name]
        n = len(b['stress'])
        same = b['stress'] == w['stress'][:n]
        if not same:
            bad = 1
            k = next(i for i, (x, y) in enumerate(zip(b['stress'], w['stress'])) if x != y) if n else 0
            print(f'{name}: OTHER, first STRESS line differing from the twin at #{k}: {b["stress"][k] if k < n else None}')
            continue
        if b['end']:
            rc = b['end'].group(1); span = float(b['end'].group(3)) - float(b['end'].group(2))
            if rc == '0' and n == len(w['stress']):
                print(f'{name}: RETURNED, {n} STRESS lines equal to the twin\'s; {w["stress"][-1]}; span {span:.1f} s')
                if name.startswith('p'):
                    issued += fast(w['stress'][-1])
            else:
                bad = 1
                print(f'{name}: OTHER, rc={rc}, {n} of {len(w["stress"])} STRESS lines; span {span:.1f} s')
            continue
        last = b['stress'][-1] if n else None
        nxt = w['stress'][n] if n < len(w['stress']) else None
        # ticks after the last heartbeat: the text after the last STRESS line
        tail = b['body'][b['body'].rfind(last) + len(last):] if last else b['body']
        after = re.findall(r'^TRAPTEST: tick ([\d.]+)$', tail, re.M)
        print(f'{name}: NO RETURN after {n} STRESS lines; last: {last}')
        print(f'  stop between {fast(last):,} and {fast(nxt):,} fast allocations (twin\'s next: {nxt})')
        print(f'  /init ticks after the last heartbeat: {len(after)}'
              + (f', last at uptime {after[-1]} (the kernel ran on: the stop is in the program)' if after
                 else ' (the console went silent: the core or kernel stopped)'))
        if b['wd']:
            print(f'  {b["wd"]} (the stage bound: "no stop within T" if the heartbeat was still advancing)')
        if name.startswith('p'):
            issued += fast(last)
    print(f'protected fast allocations completed: {issued:,}')
    sys.exit(1 if bad else 0)

main()
