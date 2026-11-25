"""kit-images-c4b.py KITDIR N : the crevoke4b bare kit's images (crevoke4b-bare-2026-11-25) in PREREG.md's "Identities and
vectors" table order, each checked before any board call: the row's sha256/16, board_rec and SR (the 88db40027 column)
must equal expected-vectors.json, images/SHA256SUMS, the .bin's own sha256 and the .rec. Exactly N rows, the control
sublet-life first. Prints NN-name:path-without-extension per image. Any mismatch exits non-zero and says what."""
import hashlib, json, os, re, sys
k, n = sys.argv[1], int(sys.argv[2])
pre = open(os.path.join(k, 'PREREG.md')).read()
rows = re.findall(r'^\| ([\w-]+) \| ([0-9a-f]{16}) \| ([0-9a-f]{8}) \| (\d+) \| [^|]+ \| (\d+) \| [^|]+ \|$', pre, re.M)
if len(rows) != n: sys.exit(f'PREREG table has {len(rows)} rows, not {n}')
if rows[0][0] != 'sublet-life': sys.exit(f'the first row is {rows[0][0]}, not the sublet-life control')
ev = json.load(open(os.path.join(k, 'expected-vectors.json')))
sums = dict(reversed(l.split()) for l in open(os.path.join(k, 'images/SHA256SUMS')) if l.strip())
if set(ev) != {r[0] for r in rows} or len(sums) != n: sys.exit('expected-vectors.json or SHA256SUMS names differ from the table')
for i, (name, sha16, rec, sr, sr9) in enumerate(rows, 1):
    b = os.path.join(k, 'images', name); e = ev[name]
    sha = hashlib.sha256(open(b + '.bin', 'rb').read()).hexdigest()
    if sha[:16] != sha16 or e['sha256'] != sha or sums.get(name + '.bin') != sha: sys.exit(f'{name}: .bin, json or SHA256SUMS sha differs from the table')
    if int(open(b + '.rec').read().strip(), 16) != int(rec, 16) or int(e['board_rec'], 16) != int(rec, 16): sys.exit(f'{name}: .rec or json board_rec differs from the table')
    if e['sr'] != int(sr) or len(e['vector']) != int(sr) or len(e['classes']) != int(sr): sys.exit(f'{name}: SR {sr} but json sr {e["sr"]}, vector {len(e["vector"])}, classes {len(e["classes"])}')
    if e['sr_904cd9c6b'] != int(sr9) or len(e['vector_904cd9c6b']) != int(sr9): sys.exit(f'{name}: base SR {sr9} differs from json')
    print(f'{i:02d}-{name}:{b}')
