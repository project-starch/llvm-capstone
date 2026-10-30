"""Write the two CheriBSD arms' cells in every case.json, from one run's records.

    fold-cheribsd.py results/<stamp>-cheribsd [--check]

The CheriBSD arms are the corpus's only ones that run-cheribsd.py records as a plain JSON run
record rather than a verdicts-v1 bundle, so derive-verdicts.py cannot write their cells. Until
2026-10-30 that meant the cells were typed by hand from the record, which is four copies of one
result and drifts exactly the way the hand-merged matrix did before derive-verdicts existed.
This does the same job for these two arms: cells in, record out, nothing typed.

One rule, applied to both arms in the same order the shared judge uses:
  attribution FIRST  -- a fault not at a site the case declared is NO-READING:unattributed,
  then the MECHANISM -- a fault the arm's mechanism did not raise is MISSED, not a catch.

The nested arm's mechanism is revocation, and under SQLITE_CONFIG_HEAP it cannot act at all,
so no fault there is ever a catch. The non-nested arm's is revocation acting on a real free,
so an attributed SIGPROT is a catch.

A fault's function comes from the run record: dladdr names it outright for a pc in a shared
library, and for a pc in the case image (every SQLite function is static, so none is in
.dynsym) the file offset pc - image_base is symbolised against that image, which carries -g.
"""
import argparse, json, os, pathlib, subprocess, sys

SELF = pathlib.Path(__file__).resolve()
CORPUS = SELF.parents[1]
REPO = SELF.parents[5]
TAGS = REPO / "capstone/ports/sqlite/repro322/tags.tsv"
OUT = pathlib.Path(os.environ.get("CHERIBSD_WORK", pathlib.Path.home() / "arms/sqlite/cheribsd")) / "out"
SYM = pathlib.Path(os.environ.get(
    "CHERI_SDK", pathlib.Path.home() / "cheriBSD/cheri/output/sdk")) / "bin/llvm-symbolizer"
OOD = ("07", "22")   # install memsys5 themselves; no non-nested form (see README)

ARMS = {
    "cheribsd-revocation": ("nested", ""),
    "cheribsd-platform": ("non-nested", "_sys"),
}


def tags():
    out = {}
    for line in TAGS.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 3 and f[2] == "case":
            out[f[1]] = f[0]
    return out


def symbolise(binary, offset):
    r = subprocess.run([str(SYM), f"--obj={binary}"], input=f"CODE 0x{offset:x}\n",
                       capture_output=True, text=True)
    name = r.stdout.splitlines()[0].strip() if r.stdout.strip() else ""
    return name if name and name != "??" else None


def where(rec, binary):
    """(function, evidence) for a faulting case, or (None, why) when it cannot be placed."""
    sym, obj = rec.get("fault_symbol"), rec.get("fault_object")
    pc, base = rec.get("fault_pc"), rec.get("image_base")
    if sym and sym != "?":
        return sym, (f"{sym} in {obj} (dladdr, object base {rec.get('fault_object_base')})")
    if pc and base and binary.is_file():
        off = int(pc, 16) - int(base, 16)
        if off >= 0:
            name = symbolise(binary, off)
            if name:
                return name, (f"{name} at {binary.name}+{off:#x} (pc {pc} less the image base "
                              f"{base} dladdr reported, symbolised against the image's own -g)")
            return None, f"pc {pc} is {off:#x} into {binary.name} and names no function there"
    return None, f"pc {pc or '?'} could not be placed (object {obj or '?'}, image base {base or '?'})"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=pathlib.Path,
                   help="the run directory, e.g. results/2026-10-30-cheribsd")
    p.add_argument("--check", action="store_true", help="report what would change, write nothing")
    a = p.parse_args()
    global REC
    REC = a.record if a.record.is_absolute() else (CORPUS / a.record)
    if not (REC / "nested").is_file() or not (REC / "non-nested").is_file():
        sys.exit(f"{REC}: needs both a `nested` and a `non-nested` run record")
    if not SYM.is_file():
        sys.exit(f"{SYM}: no llvm-symbolizer, so a fault in a case image cannot be placed. "
                 f"Set CHERI_SDK.")
    tag = tags()
    recs = {m: json.loads((REC / m).read_text()) for m in ("nested", "non-nested")}
    # Any single-case re-runs beside the two records, as context for a TIMEOUT only.
    extra = {"cases": {}}
    for side in sorted(REC.glob("non-nested-r*")):
        extra["cases"].update(json.loads(side.read_text()).get("cases", {}))
    changed, problems = [], []

    for d in sorted(CORPUS.glob("[0-9][0-9]_*")):
        p = d / "case.json"
        text = p.read_text()
        case = json.loads(text)
        before = json.dumps(case, sort_keys=True)
        sites = [s.lower() for s in (case.get("fault_sites") or [])]
        for arm, (mode, suffix) in ARMS.items():
            if arm not in (case.get("arms") or {}):
                continue
            cell = case["arms"][arm]
            rec = recs[mode]["cases"].get(d.name)
            src = f"{REC.relative_to(CORPUS)}/{mode}"
            if rec is None:
                problems.append(f"{d.name}: not in the {mode} record")
                continue
            # A TIMEOUT stays a TIMEOUT. A separate invocation of the same case that DID
            # fault is context for the note, never a substitute for the declared record:
            # choosing, among several runs, the one that reads the way the table wants is
            # the whole failure this corpus's contract exists to prevent.
            aside = ""
            if rec["verdict"] == "TIMEOUT" and d.name in extra["cases"]:
                e = extra["cases"][d.name]
                aside = (f". Run on its own against the same binary in the same boot it reports "
                         f"{e['verdict']} (si_code={e['si_code']}, {e.get('fault_symbol')} in "
                         f"{e.get('fault_object')}, a sibling single-case record, "
                         f"and again in a second such run), so this is a bound on the harness and "
                         f"not a silence of the arm -- but the declared record is this one and it "
                         f"has no reading")

            gate = (f"the run's poscontrol gate FAULTed (exit="
                    f"{recs[mode]['poscontrol_exit']}, revocation_default="
                    f"{recs[mode]['revocation_default']}) and its resolver named "
                    f"{recs[mode].get('resolver')}")
            if arm == "cheribsd-platform" and d.name[:2] in OOD:
                continue   # out-of-denominator, already written and unchanged by this run
            if rec["verdict"] == "TIMEOUT":
                v, reason, note = "NO-READING", "inconclusive", (
                    f"reached and neither completed nor faulted within the run's "
                    f"{recs[mode].get('case_timeout_s', 300)} s bound "
                    f"({' | '.join(rec.get('last') or []) or 'no output'}){aside}; {gate}")
            elif rec["verdict"] == "PASS":
                v, reason, note = "MISSED", None, (
                    f"the case RETURNED (exit {rec['exit']}) and this arm reported nothing; {gate}")
            elif rec["verdict"] == "FAULT":
                binary = OUT / f"{tag[d.name]}{suffix}"
                fn, ev = where(rec, binary)
                fault = (f"SIGPROT(34) si_code={rec['si_code']} ({rec['si_meaning']})"
                         f" pc={rec.get('fault_pc')}")
                if fn is None or fn.lower() not in sites:
                    v, reason, note = "NO-READING", "unattributed", (
                        f"{fault}: {ev}, which is not one of this case's declared "
                        f"fault_sites {case.get('fault_sites')}; {gate}")
                elif arm == "cheribsd-revocation":
                    v, reason, note = "MISSED", None, (
                        f"reached, and faulted {fault} at {ev} -- the mechanism did not reject "
                        f"the access. Revocation acts on a guest malloc free and under "
                        f"SQLITE_CONFIG_HEAP there is none, so what cleared the tag is memsys5 "
                        f"writing its Mem5Link free-list words over the block. This is a later "
                        f"consequence of an access the arm allowed, not a catch; {gate}")
                else:
                    v, reason, note = "CAUGHT", None, (
                        f"{fault}: {ev}; by function: it is one of this case's declared "
                        f"fault_sites; {gate}")
            else:
                problems.append(f"{d.name} {arm}: verdict {rec['verdict']}")
                continue
            cell["verdict"] = v
            cell["verdict_note"] = note
            cell["verdict_from"] = (
                f"{src} (--mode {mode}, {recs[mode].get('case_timeout_s', 300)} s per case)")
            cell.pop("verdict_reason", None)
            if reason:
                cell["verdict_reason"] = reason
        if json.dumps(case, sort_keys=True) != before:
            changed.append(d.name)
            if not a.check:
                p.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    for line in problems:
        print("PROBLEM", line, file=sys.stderr)
    print(f"{'would change' if a.check else 'changed'} {len(changed)} cases")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
