"""gfix-c4b.py KITDIR RUNDIR COLUMN [EDIT] : overwrite every NN-name/uart.txt of a DRY run dir with the json's COLUMN
(cand = the 88db40027 vector, base = the 904cd9c6b vector), for the grader's dry runs. EDIT: 'ctlbad' changes a sublet-life E
value; 'onebad' changes crevint's first E position; 'sent' puts the key-check sentinel in sbdrain-none; 'lat0' zeroes
crevint-lat's latencies; 'nosr' drops sbdrain-fill's SR line."""
import json, os, sys
k, rd, col = sys.argv[1:4]; edit = sys.argv[4] if len(sys.argv) > 4 else ''
ev = json.load(open(os.path.join(k, 'expected-vectors.json')))
for t in sorted(os.listdir(rd)):
    if not (len(t) > 3 and t[:2].isdigit() and os.path.isdir(os.path.join(rd, t))): continue
    name = t[3:]; e = ev[name]
    v = [int(x, 16) for x in (e['vector'] if col == 'cand' else e['vector_904cd9c6b'])]
    if edit == 'ctlbad' and name == 'sublet-life': v[3] ^= 1
    if edit == 'onebad' and name == 'b5b2-crevint': v[0] ^= 1
    if edit == 'sent' and name == 'b5b2-sbdrain-none': v[1] = 0x6b657962616420
    if edit == 'lat0' and name == 'b5b2-crevint-lat': v[12:20] = [0] * 8
    lines = ['SUPTEST BEGIN S'] + ([] if (edit == 'nosr' and name == 'b5b2-sbdrain-fill') else [f'SR {len(v):016X}']) + [f'SV {x:016X}' for x in v] + ['SUPTEST END']
    open(os.path.join(rd, t, 'uart.txt'), 'w').write('\r\n'.join(lines) + '\r\n')
print('ok')
