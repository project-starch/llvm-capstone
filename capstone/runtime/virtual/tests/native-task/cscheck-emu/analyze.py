#!/usr/bin/env python3
"""MEASUREMENT ONLY. analyze.py OUT_DIR...: per command of a run-native-stage.py run, the adapter's counter
deltas (escapes by kind, kind-3 services by route, firmware emulation entries) and the capmix window between
the command's two markers, split by where the instruction executed:
  firmware = M-mode OpenSBI, [0x8000_0000, 0x8100_0000); kernel = >= 0xffff_0000_0000_0000; user = the rest.
Exits non-zero when a window or a counter line is missing (no data is an error, not a zero)."""
import json, re, sys
from pathlib import Path

NC = 14

def snaps(path):
    out, cur = [], None
    for line in open(path):
        w = line.split()
        if not w:
            continue
        if w[0] in ('SNAP', 'FINAL'):
            cur = {'tag': w[0], 'label': int(w[2]), 'b': {}}
        elif w[0] == 'B':
            cur['b'][int(w[1], 16)] = sum(map(int, w[2:2 + NC]))
        elif w[0] == 'END':
            out.append(cur)
    return out

def cls(key):
    va = key << 24
    if va >= 0xffff000000000000:
        return 'kernel'
    if 0x80000000 <= va < 0x81000000:
        return 'firmware'
    return 'user'

def counters(s):
    return {k: int(v) for k, v in re.findall(r'(\w+)=(-?\d+)', s)}

def analyze(out):
    out = Path(out)
    log = (out / 'vm/serial.log').read_text(errors='replace').replace('\r', '')
    cnt = {(int(i), w): counters(rest) for i, w, rest in re.findall(r'^@@CNT (\d+) (begin|end) (.*)$', log, re.M)}
    rcs = {int(i): (name, rest) for i, name, rest in re.findall(r'^@@RC (\d+) (\S+) (.*)$', log, re.M)}
    s = [x for x in snaps(out / 'capmix.txt') if x['tag'] == 'SNAP']
    if not rcs or len(s) != 2 * len(rcs):
        sys.exit(f'{out}: {len(rcs)} commands, {len(s)} capmix snapshots: incomplete')
    rows = []
    for i in sorted(rcs):
        a, b = s[2 * i], s[2 * i + 1]
        assert a['label'] == 1 and b['label'] == 2, (a['label'], b['label'])
        win = {'user': 0, 'kernel': 0, 'firmware': 0}
        for k, v in b['b'].items():
            win[cls(k)] += v - a['b'].get(k, 0)
        if (i, 'begin') not in cnt or (i, 'end') not in cnt:
            sys.exit(f'{out}: command {i} has no counter lines')
        d = {k: cnt[(i, 'end')][k] - cnt[(i, 'begin')].get(k, 0) for k in cnt[(i, 'end')]}
        calls = d['register'] + d['exchange']
        exits = sum(d[f'k{k}'] for k in range(8))
        total = sum(win.values())
        rows.append(dict(command=i, name=rcs[i][0], result=rcs[i][1], instructions=win, total=total,
                         exits=exits, exits_by_kind={k: d[f'k{k}'] for k in range(8) if d[f'k{k}']},
                         register=d['register'], exchange=d['exchange'], service_other=d['service_other'],
                         delegated_calls=calls,
                         emu={k[4:]: v for k, v in d.items() if k.startswith('emu_')},
                         kernel_per_call=win['kernel'] / calls if calls else None,
                         firmware_per_call=win['firmware'] / calls if calls else None,
                         firmware_per_check=win['firmware'] / d['emu_entries'] if d.get('emu_entries') else None))
    return rows

if __name__ == '__main__':
    allrows = {}
    for out in sys.argv[1:]:
        rows = analyze(out)
        allrows[out] = rows
        for r in rows:
            print(f"{Path(out).name} cmd {r['command']} {r['name']} {r['result']}")
            print(f"  instr total {r['total']:,} user {r['instructions']['user']:,} kernel {r['instructions']['kernel']:,} firmware {r['instructions']['firmware']:,}")
            print(f"  exits {r['exits']:,} {r['exits_by_kind']}  calls {r['delegated_calls']:,} (register {r['register']:,} exchange {r['exchange']:,}) other services {r['service_other']:,}")
            fpc = r['firmware_per_check']
            print(f"  emu {r['emu']}  kernel/call {r['kernel_per_call'] or 0:,.0f} firmware/call {r['firmware_per_call'] or 0:,.0f} firmware/check {fpc or 0:,.0f}")
    Path('analysis.json').write_text(json.dumps(allrows, indent=1) + '\n') if len(sys.argv) > 1 else None
