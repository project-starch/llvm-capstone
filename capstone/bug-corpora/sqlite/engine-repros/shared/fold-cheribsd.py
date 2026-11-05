"""Fold one CheriBSD run's records into the verdicts-v1 bundle of each of its two arms.

    fold-cheribsd.py results/<stamp>-cheribsd [--check]

run-cheribsd.py writes a plain JSON run record per mode, not a bundle, so until 2026-10-31 these
two arms' cells were the only ones in the corpus that no bundle backed: first typed by hand from
the record, then written straight into case.json by this script. Both are the same defect one
layer apart -- a second writer of cells, outside the one contract every other arm is held to.
This now writes the bundles instead, and tools/derive-verdicts.py writes the cells from them, so
all seven arms of the corpus have exactly one writer and one gate.

What the two arms are (tools/arms.json carries the full text):
  cheribsd-revocation -> cheribsd-sqlite-memsys5   memsys5 over one static array; catch_causes []
  cheribsd-platform   -> cheribsd-sqlite-sysalloc  the arena removed; catch_causes [1, 2]
`catch_causes` here holds FreeBSD SIGPROT si_codes, NOT Capstone fault causes; nothing but that
key reads the number, and the two namespaces must never be compared.

The judge is tools/verdicts.py and it is not reimplemented here. This only turns each recorded
case into an Observation -- reached, completed, where it faulted, and what ties that to the
defect -- and hands it over. Every field comes from the record; what the run did not record is
left out rather than reconstructed, which is why image_sha256 is None on every row and the
images' hashes sit in inputs.json with the basis on which they were taken after the run.

A fault's function comes from the record: dladdr names it outright for a pc in a shared library,
and for a pc in the case image (every SQLite function is static, so none reaches .dynsym) the
file offset pc - image_base is symbolised against that image, which carries -g. When dladdr
resolved NOTHING it prints `?` and 0x0, and that is a failure sentinel, not an address -- see
placed() for the cell this cost.
"""
import argparse, json, os, pathlib, subprocess, sys

SELF = pathlib.Path(__file__).resolve()
CORPUS = SELF.parents[1]
REPO = SELF.parents[5]
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
import verdicts as v  # noqa: E402

TAGS = REPO / "capstone/ports/sqlite/repro322/tags.tsv"
OUT = pathlib.Path(os.environ.get("CHERIBSD_WORK", pathlib.Path.home() / "arms/sqlite/cheribsd")) / "out"
SDK = pathlib.Path(os.environ.get("CHERI_SDK", pathlib.Path.home() / "cheriBSD/cheri/output/sdk"))
GUEST = SDK.parent / "rootfs-riscv64-purecap/boot/kernel/kernel"
SYM = SDK / "bin/llvm-symbolizer"
OOD = ("07", "22")   # install memsys5 themselves; no non-nested form (see README)

# mode -> (arm name, case-binary suffix, configuration in tools/arms.json)
ARMS = {
    "nested": ("cheribsd-revocation", "", "cheribsd-sqlite-memsys5"),
    "non-nested": ("cheribsd-platform", "_sys", "cheribsd-sqlite-sysalloc"),
}
# The run that this fold's platform hashes must predate, so that hashing a file now measures the
# bytes that ran. Each artifact is checked against the EARLIEST record of the run, not its own
# mode's, because run-cheribsd.py builds before it runs any case.
PLATFORM = {
    "compiler": SDK / "bin/clang",
    "qemu": SDK / "bin/qemu-system-riscv64cheri",
    "firmware": SDK / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin",
    "kernel": GUEST,
    "runner": CORPUS / "shared/run-cheribsd.py",
}
# The judge is deliberately NOT one of those: a bundle exists so a stored Observation can be
# re-judged, so this hash records which judge wrote these verdicts -- today's -- and a judge
# newer than the run is the normal case, not a provenance error. The first run of this fold
# stopped on exactly that, which is the guard above working.
JUDGE = TOOLS / "verdicts.py"
# Pre-registered before the run, on the lane branch, so the prediction cannot be fitted to it.
PREREGISTRATION = "18d692d16529"


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


def placed(rec, binary):
    """(function, evidence) for a faulting case, or (None, why) when it cannot be placed."""
    sym, obj = rec.get("fault_symbol"), rec.get("fault_object")
    pc, base, image = rec.get("fault_pc"), rec.get("image_base"), rec.get("image")
    if sym and sym != "?":
        return sym, f"{sym} in {obj} (dladdr, object base {rec.get('fault_object_base')})"
    if not pc:
        return None, "the record has no fault pc"
    # dladdr prints `?` and 0x0 when it resolved NOTHING, and that is a sentinel, not an address.
    # Until 2026-10-31 this fold subtracted the 0x0 as if it were a load base and symbolised the
    # pc itself as a file offset into the case image. For case 22 that landed on a plausible
    # name, editDist3FindLang, which IS one of its declared fault_sites, and the cell read MISSED
    # on an arm where a MISSED is a measurement. The pc is not even inside the image -- 0x6a2ba,
    # below the 0x100000 base every other image reports. The run's poscontrol gate cannot catch
    # this: it proves dladdr works in the GATE's process, not that it resolved in each case's.
    if not image or image == "?" or not base or int(base, 16) == 0:
        return None, (f"dladdr resolved nothing for pc {pc}: it reported object {obj!r}, image "
                      f"{image!r} and image base {base!r}, which is its failure sentinel rather "
                      f"than an address, so nothing in the record places this fault")
    off = int(pc, 16) - int(base, 16)
    if off < 0:
        return None, (f"pc {pc} lies BELOW the image base {base} dladdr reported, so it is not "
                      f"an offset into {binary.name}")
    if not binary.is_file():
        return None, f"pc {pc} is {off:#x} into {binary.name}, which is not on this host"
    name = symbolise(binary, off)
    if not name:
        return None, f"pc {pc} is {off:#x} into {binary.name} and names no function there"
    return name, (f"{name} at {binary.name}+{off:#x} (pc {pc} less the image base {base} dladdr "
                  f"reported, symbolised against the image's own -g)")


def gate(record):
    """The run's poscontrol control, as one Control attached to every Observation of the run."""
    code, res = record["poscontrol_exit"], record.get("resolver")
    observed = "fault" if code == "162" else "complete" if code == "0" else "none"
    return v.Control("poscontrol", observed, (
        f"ports/sqlite/cheribsd/poscontrol.c, a use-after-free through the SYSTEM allocator, run "
        f"in this boot before the cases: exit={code}, the guest's "
        f"vm.cheri_revoke_every_free default was {record['revocation_default']}, and its own "
        f"fault handler resolved {res}"))


def observe(case, name, arm, mode, rec, sites, binary, record):
    """One Observation, from the record and nothing else."""
    o = v.Observation(case=name, arm=arm, controls=[gate(record)])
    if arm == "cheribsd-platform" and name[:2] in OOD:
        # Its own run_case() installs memsys5 over sqlite_heap, so it never reaches
        # repro_init()'s NOMEM5 branch and what ran was the nested configuration under the
        # stock quarantine. The instrument is the run's own marker, with 31 positives: 31 of
        # the 33 cases record allocator=system AND revoke=on-every-free, and exactly 07 and 22
        # record neither. The 2026-10-29 run reads the same way.
        o.infra = "out-of-denominator"
        o.notes = (f"this run records neither allocator=system nor revoke=on-every-free for the "
                   f"case (31 of its 33 cases record both; only 07 and 22 record neither), "
                   f"because its run_case() installs memsys5 over sqlite_heap itself and so "
                   f"never reaches repro_init()'s NOMEM5 branch. What ran was the nested "
                   f"configuration under the stock quarantine, not this arm, so its reading is "
                   f"not a reading of this arm. The Capstone side agrees: the builder reports "
                   f"the case not applicable without memsys5")
        return o
    if rec["verdict"] == "TIMEOUT":
        o.infra = "inconclusive"
        o.notes = (f"neither completed nor faulted within the run's {record['case_timeout_s']} s "
                   f"bound ({' | '.join(rec.get('last') or []) or 'no output'}). A separate "
                   f"invocation that did fault is context for this note and never a substitute: "
                   f"choosing, among several runs, the one that reads the way the table wants is "
                   f"the whole failure this contract exists to prevent")
        return o
    if rec["verdict"] == "PASS":
        o.reached, o.completed = True, True
        o.reach_evidence = (f"the case ran to its own end and returned 0 "
                            f"({' | '.join(rec.get('last') or [])})")
        return o
    if rec["verdict"] != "FAULT":
        raise ValueError(f"{name} {arm}: the record's verdict is {rec['verdict']!r}")
    fn, ev = placed(rec, binary)
    o.reached = True
    o.reach_evidence = (f"the case faulted inside its own run, after the harness entered it "
                        f"({(rec.get('last') or ['no output'])[0]})")
    o.fault = v.Fault(cause=int(rec["si_code"]), pc=int(rec["fault_pc"], 16),
                      symbol=fn, offset=None)
    o.notes = f"SIGPROT(34) si_code={rec['si_code']} ({rec['si_meaning']})"
    if fn is not None and fn.lower() in sites:
        o.attribution = "function"
        o.attribution_evidence = f"{ev}, one of this case's declared fault_sites"
    else:
        o.attribution_evidence = (
            f"{ev}; it is not one of this case's declared fault_sites "
            f"{case.get('fault_sites')}" if fn is not None else ev)
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=pathlib.Path, help="the run directory, e.g. results/2026-10-30-cheribsd")
    p.add_argument("--check", action="store_true", help="report what would change, write nothing")
    a = p.parse_args()
    rec_dir = a.record if a.record.is_absolute() else (CORPUS / a.record)
    missing = [m for m in ARMS if not (rec_dir / m).is_file()]
    if missing:
        sys.exit(f"{rec_dir}: needs a run record per mode; missing {missing}")
    if not SYM.is_file():
        sys.exit(f"{SYM}: no llvm-symbolizer, so a fault in a case image cannot be placed. "
                 f"Set CHERI_SDK.")
    records = {m: json.loads((rec_dir / m).read_text()) for m in ARMS}
    cutoff = min((rec_dir / m).stat().st_mtime for m in ARMS)
    for mode, r in records.items():
        if r["poscontrol_exit"] != "162":
            sys.exit(f"{mode}: the run's gate did not fault (exit={r['poscontrol_exit']}); "
                     f"the run is void and must not become a bundle")
        if not isinstance(r.get("resolver"), dict) or not r["resolver"].get("base"):
            sys.exit(f"{mode}: the record does not say the gate's dladdr resolver fired "
                     f"({r.get('resolver')!r}), so no fault in it can be placed")
        if r.get("skipped"):
            sys.exit(f"{mode}: the run skipped {sorted(r['skipped'])}; a partial run is not a bundle")

    # Platform hashes. A provenance hash must come from the MEASUREMENT, not from the host's
    # state whenever the fold is next run, so each one is taken once -- from a file that still
    # predates the run -- and then read back out of the bundle. A file that has since changed is
    # no longer re-verifiable, which is not the same as wrong: the recorded hash stands and the
    # fold says which keys it could not re-check. Only a DISAGREEMENT on a file that still
    # predates the run is an error, and that is the case this guard exists for.
    platform, stale, wrong = {}, [], []
    recorded = {}
    for arm in (a for a, _, _ in ARMS.values()):
        f = rec_dir / arm / "inputs.json"
        if f.is_file():
            recorded = json.loads(f.read_text()).get("platform", {})
            break
    for key, path in PLATFORM.items():
        real = path.resolve()
        if real.is_file() and real.stat().st_mtime <= cutoff:
            platform[key] = v.sha256(real)
            if key in recorded and recorded[key] != platform[key]:
                wrong.append(f"{key}: the bundle records {recorded[key][:16]}..., {real} hashes "
                             f"{platform[key][:16]}... and still predates the run")
        elif key in recorded:
            platform[key] = recorded[key]
            stale.append(key)
        else:
            why = "newer than the run" if real.is_file() else "not on this host"
            sys.exit(f"{key}: {path} is {why} and no bundle records it, so nothing can say "
                     f"what ran")
    if wrong:
        sys.exit("the recorded provenance disagrees with the host:\n  " + "\n  ".join(wrong))
    if stale:
        print(f"note: {', '.join(sorted(stale))} changed on this host since the run; the "
              f"bundle's own recorded hash is kept", file=sys.stderr)
    platform["judge"] = v.sha256(JUDGE)
    platform["profile"] = "cheribsd-riscv64-purecap"
    platform["hashed"] = (
        "2026-10-31, after the run: run-cheribsd.py of 2026-10-30 recorded no hashes. Each was "
        "taken from a file whose mtime precedes the run's first record, so the bytes hashed are "
        "the bytes that ran; a later fold reads it back from here rather than re-hashing a host "
        "that has moved on, and stops if a file that still predates the run disagrees. kernel is "
        "the purecap kernel installed in the guest rootfs; the disk image itself is NOT hashed "
        "because the guest writes to it. judge is always the current one, by design")

    tag, problems, bundles = tags(), [], {}
    cases = sorted(CORPUS.glob("[0-9][0-9]_*"))
    for mode, (arm, suffix, config) in ARMS.items():
        record, rows, images = records[mode], [], {}
        for d in cases:
            case = json.loads((d / "case.json").read_text())
            if arm not in (case.get("arms") or {}):
                continue
            r = record["cases"].get(d.name)
            if r is None:
                problems.append(f"{d.name}: not in the {mode} record")
                continue
            binary = OUT / f"{tag[d.name]}{suffix}"
            sites = [s.lower() for s in (case.get("fault_sites") or [])]
            o = observe(case, d.name, arm, mode, r, sites, binary, record)
            if binary.is_file() and binary.stat().st_mtime <= cutoff:
                images[d.name] = v.sha256(binary)
            rows.append((o, v.judge(o, v.load_arms()[config])))
        inputs = {
            "configuration": config,
            "mode": mode,
            "platform": platform,
            "record": str((rec_dir / mode).relative_to(CORPUS)),
            "case_timeout_s": record["case_timeout_s"],
            "predicted": (f"0 CAUGHT on every case: the mechanism acts on a libc free and under "
                          f"SQLITE_CONFIG_HEAP there is none" if mode == "nested" else
                          f"CAUGHT on every case the arm is in the denominator for"),
            "predicted_where": f"pre-registered on the lane branch in {PREREGISTRATION}, before the run",
            "images": {
                "sha256": images,
                "basis": platform["hashed"],
                "note": ("not on the rows: the Observation is what the run observed, and this run "
                         "observed no image hash. run-cheribsd.py records image_sha256 per case "
                         "from 2026-10-31 on, and a later fold takes it from the record"),
            },
        }
        bundles[arm] = (rec_dir / arm, rows, inputs)

    if problems:
        for line in problems:
            print("PROBLEM", line, file=sys.stderr)
        return 1
    for arm, (out, rows, inputs) in sorted(bundles.items()):
        tally = {}
        for _, (verdict, reason, _) in rows:
            key = verdict if verdict != v.NO_READING else f"{verdict}:{reason}"
            tally[key] = tally.get(key, 0) + 1
        if a.check:
            was = (out / "inputs.json")
            if not was.is_file():
                print(f"{out.name}: WOULD BE WRITTEN  {len(rows)} rows {dict(sorted(tally.items()))}")
                continue
            old = json.loads(was.read_text())
            same = (old.get("tally") == dict(sorted(tally.items())) and old.get("cases") == len(rows))
            print(f"{out.name}: {'current' if same else 'WOULD CHANGE'}  "
                  f"{len(rows)} rows {dict(sorted(tally.items()))}")
        else:
            v.write_bundle(out, "sqlite/engine-repros", arm, rows, inputs)
            print(f"{out.name}: {len(rows)} rows {dict(sorted(tally.items()))} -> "
                  f"{out.relative_to(CORPUS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
