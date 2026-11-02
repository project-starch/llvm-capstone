#!/usr/bin/env python3
"""Re-derive this record's README from matrix.tsv. It READS the README's own numbers and compares
them with the matrix; it does not compare the matrix against constants kept here, because a checker
that holds its own copy of the answer passes whatever the README says. Negative-tested: change one
digit in the README and this exits 1."""
import collections, csv, json, re, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
rows = list(csv.DictReader(open(R/"matrix.tsv"), delimiter="\t"))
md = (R/"README.md").read_text()
TEMPORAL = ("CWE-416", "CWE-415", "CWE-590")
tmp = [r for r in rows if r["cwe"] in TEMPORAL]
fail = 0
def chk(ok, what):
    global fail
    print(("  OK   " if ok else "  FAIL ") + what); fail += (not ok)

a = collections.Counter(r["release_destination"] for r in rows)
t = collections.Counter(r["release_destination"] for r in tmp)

# --- the result table: every row's two counts, as the README prints them ---
seen = set()
for cls, all70, t55 in re.findall(r"^\| `([a-zA-Z-]+)` \| (\d+) \| \*?\*?(\d+)\*?\*? \|", md, re.M):
    seen.add(cls)
    chk(a[cls] == int(all70), f"result table, {cls}, all 70: README {all70}, matrix {a[cls]}")
    chk(t[cls] == int(t55), f"result table, {cls}, temporal: README {t55}, matrix {t[cls]}")
chk(seen == set(a), f"the table lists exactly the classes the matrix holds: {sorted(seen)}")
chk(len(rows) == 70 and len(tmp) == 55, f"70 rows, 55 temporal ({len(rows)}/{len(tmp)})")

# --- the sentences that carry a number ---
m = re.search(r"\*\*(\d+) of the (\d+) temporal cases are invisible", md)
chk(m and int(m.group(1)) == t["nested"] + t["nested-until-teardown"] and int(m.group(2)) == len(tmp),
    f"'N of the 55 temporal are invisible': README {m.group(1) if m else '?'}, "
    f"matrix {t['nested'] + t['nested-until-teardown']}")
m = re.search(r"(\d+) of the 19 are not temporal defects", md)
nont = sum(1 for r in rows if r["release_destination"] == "no-release-observed" and r["cwe"] not in TEMPORAL)
chk(m and int(m.group(1)) == nont, f"'N of the 19 are not temporal': README {m.group(1) if m else '?'}, matrix {nont}")

# --- the axis table: each cell's counts, per population, as the README prints them ---
for field, label in (("population_upstream", "upstream corpus (gcc 13)"),
                     ("population_here", "this corpus's run (gcc 15.2)")):
    line = next((l for l in md.splitlines() if l.startswith("| the " + label[:8])
                 or l.startswith("| this corpus") and field == "population_here"), None)
    line = next(l for l in md.splitlines() if label in l)
    g = collections.Counter((r[field], r["release_destination"]) for r in tmp)
    for pop, cell in zip(("asan-blind", "temporal-asan-visible"), line.split("|")[2:4]):
        got = {k[1]: v for k, v in g.items() if k[0] == pop}
        chk(sum(got.values()) == int(re.search(r"n=(\d+)", cell).group(1)),
            f"{label} / {pop}: n= in README vs matrix total {sum(got.values())}")
        # normalise the cell first: the README emphasises and hedges ("**9 in fact `non-nested`**"),
        # and a loose pattern otherwise reads the "n=22" total as one of the counts.
        flat = cell.replace("*", "").replace("`", "").replace(" in fact ", " ")
        for name, key in (("nested", "nested"), ("until-teardown", "nested-until-teardown"),
                          ("not-Zend", "not-Zend"), ("non-nested", "non-nested"),
                          ("unobserved", "no-release-observed")):
            pat = r"(?<![=\d])(\d+) " + re.escape(name) + r"\b"
            if key == "nested":            # never let non-nested or -until-teardown match here
                pat = r"(?<![=\d])(\d+) (?<!non-)nested(?!-)"
            m = re.search(pat, flat)
            want = got.get(key, 0)
            chk((int(m.group(1)) if m else 0) == want,
                f"{label} / {pop} / {key}: README {m.group(1) if m else 0}, matrix {want}")

# --- the cases the README names must be the cases the matrix finds misplaced ---
seg = md.split("wrong on nine of them")[1].split("Eight are")[0]
named = sorted({"CRASH-" + n for n in re.findall(r"(?:CRASH-)?-?(\d{3})\b", seg)})
wrong = sorted(r["source_id"] for r in tmp if r["population_here"] == "asan-blind"
               and r["release_destination"] in ("non-nested", "nested-until-teardown", "not-Zend"))
chk(named == wrong, f"the cases the README names are the ones the matrix finds\n"
                    f"         matrix={wrong}\n         named ={named}")
# --- the corpus README's by-release section quotes rates from these same bundles ---
corpus_md = R.parent.parent / "README.md"
if corpus_md.is_file() and "split by where the release was MEASURED to go" in corpus_md.read_text():
    cmd = corpus_md.read_text()
    cls = {r["source_id"]: r["release_destination"] for r in rows}
    ids = {d.name: json.loads((d/"oracle.json").read_text())["id"]
           for d in sorted((R.parent.parent).iterdir()) if (d/"oracle.json").is_file()}
    pop = {d.name: json.loads((d/"oracle.json").read_text()) for d in sorted((R.parent.parent).iterdir())
           if (d/"oracle.json").is_file()}
    def bundle(path):
        return {json.loads(l)["observation"]["case"]: json.loads(l)
                for l in (R.parent.parent/path).read_text().splitlines() if l.strip()}
    base = bundle("results/2026-10-28-run18/sensitivity/bundle-virtual-malloc/verdicts.jsonl")
    ne = bundle("results/2026-10-28-run18/sensitivity/bundle-virtual-nested-exact/verdicts.jsonl")
    s48 = [c for c in ids if pop[c]["cwe"] in TEMPORAL and ne[c]["verdict"] == "CAUGHT"]
    def rate(sel):
        hit = sum(1 for c in sel if base[c]["verdict"] == "CAUGHT")
        return hit, len(sel), round(100 * hit / len(sel)) if sel else 0
    grp = {"reach": [c for c in s48 if cls[ids[c]] in ("non-nested", "not-Zend")],
           "park":  [c for c in s48 if cls[ids[c]] in ("nested", "nested-until-teardown")]}
    for key, phrase in (("reach", r"\*\*(\d+) of the (\d+)\*\* whose block reaches libc"),
                        ("park",  r"\*\*(\d+) of the (\d+)\*\* whose block does not")):
        m = re.search(phrase, cmd)
        h, n, _ = rate(grp[key])
        chk(m and (int(m.group(1)), int(m.group(2))) == (h, n),
            f"corpus README, baseline on the {key} side: README "
            f"{m.groups() if m else '?'}, bundles ({h}, {n})")
    m = re.search(r"(\d+) % against (\d+) %\. Split", cmd)
    chk(m and (int(m.group(1)), int(m.group(2))) == (rate(grp["reach"])[2], rate(grp["park"])[2]),
        f"corpus README, measured-axis rates: README {m.groups() if m else '?'}, "
        f"bundles ({rate(grp['reach'])[2]}, {rate(grp['park'])[2]})")
    for field, pat in (("population", r"reads (\d+) % against (\d+) %, which does not tell"),
                       ("collaborator_population", r"between them, (\d+) % against (\d+) %")):
        vis = [c for c in s48 if pop[c].get(field) == "temporal-asan-visible"]
        bld = [c for c in s48 if pop[c].get(field) == "asan-blind"]
        m = re.search(pat, cmd)
        chk(m and (int(m.group(1)), int(m.group(2))) == (rate(vis)[2], rate(bld)[2]),
            f"corpus README, {field} rates: README {m.groups() if m else '?'}, "
            f"bundles ({rate(vis)[2]}, {rate(bld)[2]})")
    # the README now rests "48 of 48 is not an artefact" on the arm having no MISSED at all
    m = re.search(r"the\s+arm has \*\*(\d+) MISSED\*\*", cmd)
    nmiss = sum(1 for c in ids if pop[c]["cwe"] in TEMPORAL and ne[c]["verdict"] == "MISSED")
    chk(m and int(m.group(1)) == nmiss,
        f"corpus README, nested exact MISSED over the temporal cases: "
        f"README {m.group(1) if m else '?'}, bundles {nmiss}")
    # parse the WORD, so changing it is caught; counting occurrences of a fixed word is not a check
    WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    m = re.search(r"the other (\w+)\s+returned none", cmd)
    nread = sum(1 for c in ids if pop[c]["cwe"] in TEMPORAL and ne[c]["verdict"] != "CAUGHT")
    chk(m is not None and WORDS.get(m.group(1).lower()) == nread,
        f"corpus README, the no-readings the text names: README {m.group(1) if m else '?'}, "
        f"bundles {nread}")
    sp = sum(1 for c in ids if pop[c]["cwe"] in ("CWE-125", "CWE-787"))
    m = re.search(r"all (\d+) spatial cases read `no release observed`", cmd)
    chk(m and int(m.group(1)) == sp and all(cls[ids[c]] == "no-release-observed"
                                            for c in ids if pop[c]["cwe"] in ("CWE-125", "CWE-787")),
        f"corpus README, spatial count: README {m.group(1) if m else '?'}, corpus {sp}")

print("\nALL README NUMBERS RE-DERIVED FROM matrix.tsv" if not fail else f"\n{fail} MISMATCH(ES)")
sys.exit(1 if fail else 0)
