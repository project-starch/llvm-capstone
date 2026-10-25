"""Compare the native gate file by file: TAP plan and every ok / not ok line (number, text,
directive), the harness's Result line, and the raw same-answers outputs byte for byte."""
import hashlib, json, re, sys
from pathlib import Path
G = Path(sys.argv[1])
A, B = G / "without-0011", G / "with-0011"
TAP = re.compile(r"^(?:(?:not )?ok\b.*|1\.\.\d+.*|Bail out!.*)$")
def tap(p):
    t = p.read_text(errors="replace")
    lines = [l.rstrip() for l in t.splitlines() if TAP.match(l.strip())]
    res = re.search(r"^Result: (\w+)", t, re.M)
    oks = sum(1 for l in lines if l.startswith("ok"))
    nok = sum(1 for l in lines if l.startswith("not ok"))
    plan = next((l for l in lines if l.startswith("1..")), "")
    return dict(lines=lines, result=res.group(1) if res else None, ok=oks, not_ok=nok, plan=plan)
rows, same_all = [], True
for f in sorted((A / "harness").iterdir()):
    a, b = tap(f), tap(B / "harness" / f.name)
    same = a["lines"] == b["lines"] and a["result"] == b["result"]
    same_all &= same
    rows.append(dict(file=f.stem.replace("_", "/", 1), without=dict((k, a[k]) for k in ("plan", "ok", "not_ok", "result")),
                     with_0011=dict((k, b[k]) for k in ("plan", "ok", "not_ok", "result")), identical_tap=same))
raw = {}
for f in sorted((A / "raw").iterdir()):
    x, y = f.read_bytes(), (B / "raw" / f.name).read_bytes()
    raw[f.name] = dict(identical=x == y, sha256_without=hashlib.sha256(x).hexdigest(), sha256_with=hashlib.sha256(y).hexdigest())
    same_all &= x == y
json.dump(dict(harness=rows, raw=raw, all_identical=same_all), sys.stdout, indent=1)
