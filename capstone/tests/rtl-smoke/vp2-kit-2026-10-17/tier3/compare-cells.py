#!/usr/bin/env python3
# compare-cells.py <manifest.json> <corpus> <arm> <verdicts.jsonl> : each new cell against its recorded cell by
# (verdict, fault cause, fault symbol, attribution, reached). Exit 2 if nothing compared or a case is missing.
import json, sys
man, corpus, arm, vj = sys.argv[1:5]
m = json.load(open(man))
cells = [c for c in m['cells'] if c['corpus'] == corpus and c['arm'] == arm]
new = {}
for l in open(vj):
    j = json.loads(l); new[j['observation']['case']] = j
k = lambda o: ((o.get('fault') or {}).get('cause'), (o.get('fault') or {}).get('symbol'), o.get('attribution'), o.get('reached'))
if not cells or not new: sys.exit(f"NO DATA: recorded {len(cells)} new {len(new)}")
bad, missing = [], []
for c in cells:
    case = c['case']; n = new.get(case)
    if n is None: missing.append(case); continue
    if (c['verdict'], c.get('reason'), k(c['observation'])) != (n['verdict'], n.get('reason'), k(n['observation'])):
        bad.append((case[:30], (c['verdict'], c.get('reason'), k(c['observation'])), (n['verdict'], n.get('reason'), k(n['observation']))))
print(f"{corpus} {arm}: recorded {len(cells)} new {len(new)} mismatches {len(bad)} missing {len(missing)}")
for b in bad: print("  DIFF", b)
sys.exit(2 if missing else 0)
